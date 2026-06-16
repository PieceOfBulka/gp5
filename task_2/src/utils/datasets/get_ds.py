import os
import pandas as pd
from clearml import Dataset

def get_dataset(ds_name: str, ds_project:str) -> pd.DataFrame:
    path = Dataset.get(
        dataset_project=ds_project,
        dataset_name=ds_name
    ).get_local_copy()

    return pd.read_csv(filepath_or_buffer=os.path.join(path, os.listdir(path)[0]))