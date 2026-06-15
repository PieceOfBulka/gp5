# tf-idf docs: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# logreg docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import logging
import pandas as pd
import ast
import os
import joblib

from clearml import Dataset, Task


logging.basicConfig(level=logging.INFO)

task = Task.init(project_name="HSE-GP5", task_name='TF-IDF+LogReg+OneVSall+class_weight baseline')

configuration={'max_features':5000, 'ngram_range':(1,2), 'stop_words':'english', 'model_state':42, 'test_size':0.2, 'split_state':42}
task.connect(configuration)

logging.info('Скачиваем датасет...')
dataset = Dataset.get(dataset_project="HSE-GP5", dataset_name='final_lyrics_dataset')
dataset_path = dataset.get_local_copy()
df = pd.read_csv(os.path.join(dataset_path,'final_dataset.csv'),encoding='utf-8',escapechar="\\")
df = df[['lyrics','genres']].dropna()
df['genres'] = df['genres'].apply(ast.literal_eval)

logging.info("Переводим списки жанров в мультилейблы...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])
task.upload_artifact(name="genre_classes",artifact_object=set(mlb.classes_))

logging.info('Разбиваем на train/test...')
Xtrain, Xtest, ytrain, ytest = train_test_split(df['lyrics'],y,test_size=configuration['test_size'],random_state=configuration['split_state'])

logging.info('Прогоняем текста песен через TF-IDF...')
tf = TfidfVectorizer(max_features=configuration['max_features'],ngram_range=configuration['ngram_range'],stop_words=configuration['stop_words'])
Xtrain_tfidf = tf.fit_transform(Xtrain)
Xtest_tfidf = tf.transform(Xtest)

logging.info('Обучаем LogRegи методом OneVSall...')
model = OneVsRestClassifier(LogisticRegression(random_state=configuration['model_state'],verbose=1,class_weight="balanced"))
model.fit(Xtrain_tfidf,ytrain)

logging.info('Делаем предсказания...')
predictions = model.predict(Xtest_tfidf)

task.get_logger().report_single_value(name='accuracy', value=accuracy_score(ytest,predictions))
task.get_logger().report_single_value(name='f1 macro', value=f1_score(ytest,predictions,average='macro'))
task.get_logger().report_single_value(name='f1 micro', value=f1_score(ytest,predictions,average='micro'))
task.upload_artifact(name='classification_report',artifact_object=classification_report(ytest,predictions,target_names=mlb.classes_))

logging.info('Сохраняем модель...')
model_path='onevsall_tfidf.kl'
joblib.dump(model, model_path)
task.update_output_model(model_path=model_path,model_name='TF-IDF_OneVSall_baseline')

logging.info('Результаты сохранены!')
task.close()