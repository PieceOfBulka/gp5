from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from torch import nn
import torch

import logging
import numpy as np
from functools import partial

from clearml import Task
from transformers import AutoTokenizer

from lstm_class import BiLSTMClassificationModel
from get_dataset import load_dataset, LyricsDataset, collate_batch


logging.basicConfig(level=logging.INFO)

task = Task.init(project_name="HSE-GP5", task_name='BiLSTM, epochs=20 + threshold=0.15')


configuration={'test_size':0.2,
               'split_state':42,
               'max_length':256,
               'batch_size':32,
               'embed_dim':128,
               'hidden_dim':192,
               'dropout':0.3,
               'epohs_number':20,
               'learning_rate':5e-4,
               'threshold':0.15,
               'weight_decay':1e-4}
task.connect(configuration)

logging.info('Скачиваем датасет...')
df = load_dataset(dataset_project="HSE-GP5", dataset_name='final_lyrics_dataset')

logging.info("Переводим списки жанров в мультилейблы...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])
task.upload_artifact(name="genre_classes",artifact_object=set(mlb.classes_))

logging.info('Разбиваем на train/test...')
Xtrain, Xtest, ytrain, ytest = train_test_split(df['lyrics'],y,test_size=configuration['test_size'],random_state=configuration['split_state'])

logging.info('Скачиваем токенизатор...')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

logging.info('Формируем train+test датасеты')
collate=partial(collate_batch,tokenizer=tokenizer,max_length=configuration['max_length'])
train = DataLoader(LyricsDataset(Xtrain,ytrain),batch_size=configuration['batch_size'],collate_fn=collate,shuffle=True)
test = DataLoader(LyricsDataset(Xtest,ytest),batch_size=configuration['batch_size'],collate_fn=collate)



logging.info('Инициализируем Bi-LSTM...')
model = BiLSTMClassificationModel(tokenizer.vocab_size,
                                  embed_dim=configuration['embed_dim'],
                                  hidden_dim=configuration['hidden_dim'],
                                  num_classes=len(mlb.classes_),
                                  pad_token_id=tokenizer.pad_token_id,
                                  dropout=configuration['dropout'])

logging.info('Начинаем обучение Bi-LSTM...')
positive=ytrain.sum(axis=0)
negative=ytrain.shape[0]-positive
criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((negative+1)/(positive+1),1,10),dtype=torch.float32))
optimizer=torch.optim.Adam(model.parameters(),lr=configuration['learning_rate'],weight_decay=configuration['weight_decay'])

for epoch in range(configuration['epohs_number']):
    logging.info(f"Эпоха {epoch+1}/{configuration['epohs_number']}")
    model.train()
    train_loss=0.0
    train_probs=[]
    train_targets=[]
    count=0
    for ids,mask,y in train:
        optimizer.zero_grad()
        predictions=model(ids,mask)
        loss=criterion(predictions,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        count+=1
        if count%10==0:
            logging.info(f'Train loss на шаге {count} = {np.round(loss.item(),4)}')

        train_probability=torch.sigmoid(predictions)
        train_probs.append((train_probability>configuration['threshold']).numpy())
        train_targets.append(y.numpy())
    p_train=np.vstack(train_probs)
    t_train=np.vstack(train_targets)
    
    task.get_logger().report_scalar(title='loss', value=train_loss/len(train), series='train', iteration=epoch+1)
    task.get_logger().report_scalar(title='accuracy', value=accuracy_score(t_train,p_train), series='train', iteration=epoch+1)
    task.get_logger().report_scalar(title='f1 macro', value=f1_score(t_train,p_train,average='macro'), series='train', iteration=epoch+1)
    task.get_logger().report_scalar(title='f1 micro', value=f1_score(t_train,p_train,average='micro'), series='train', iteration=epoch+1)

    logging.info(f'Считаем validation метрики для эпохи {epoch+1}')
    model.eval()
    probabilities=[]
    targets=[]
    val_loss=0.0
    with torch.no_grad():
        for ids,mask,y in test:
            preds=model(ids,mask)
            loss=criterion(preds,y)
            val_loss+=loss.item()
            probability=torch.sigmoid(preds)
            probabilities.append((probability>configuration['threshold']).numpy())
            targets.append(y.numpy())
    p=np.vstack(probabilities)
    t=np.vstack(targets)
    task.get_logger().report_scalar(title='loss', value=val_loss/len(test), series='validation', iteration=epoch+1)
    task.get_logger().report_scalar(title='accuracy', value=accuracy_score(t,p), series='validation', iteration=epoch+1)
    task.get_logger().report_scalar(title='f1 macro', value=f1_score(t,p,average='macro'), series='validation', iteration=epoch+1)
    task.get_logger().report_scalar(title='f1 micro', value=f1_score(t,p,average='micro'), series='validation', iteration=epoch+1)
    logging.info(f'Validation loss = {val_loss}')

logging.info('Сохраняем модель...')
model_path='bilstm.pt'
torch.save(model.state_dict(), model_path)
task.update_output_model(model_path=model_path,model_name='bilstm (20 epochs) + threshold=0.15')

logging.info('Результаты сохранены!')
task.close()