# код взят из семинарского ноутбука
import pandas as pd
import os
import ast
import numpy as np

import torch
from torch.utils.data import Dataset
from clearml import Dataset as clearmldataset


class LyricsDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, ind):
        return self.X.iloc[ind], self.y[ind]


def load_dataset(dataset_project,dataset_name):
    dataset = clearmldataset.get(dataset_project=dataset_project, dataset_name=dataset_name)
    dataset_path = dataset.get_local_copy()
    df = pd.read_csv(os.path.join(dataset_path,'final_dataset.csv'),encoding='utf-8',escapechar="\\")
    df = df[['lyrics','genres']].dropna()
    df['genres'] = df['genres'].apply(ast.literal_eval)
    return df


def collate_batch(batch, tokenizer, max_length):
    text_list, genre_list = [], []
    for _text, _label, in batch:
        text_list.append(_text)
        genre_list.append(_label)

    encoded = tokenizer(text_list,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
    return encoded['input_ids'], encoded['attention_mask'], torch.tensor(np.array(genre_list),dtype=torch.float32)