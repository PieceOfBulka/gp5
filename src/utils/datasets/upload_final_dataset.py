import os
import zipfile
from clearml import StorageManager, Dataset
from pathlib import Path
base_dir=Path(__file__).resolve().parent.parent.parent.parent

dataset = Dataset.create(dataset_project="HSE-GP5", dataset_name='final_lyrics_dataset')    

dataset.add_files(path=base_dir / 'dataset_forming/data/final_dataset.csv')

dataset.upload()

dataset.finalize()