import os
import zipfile
from clearml import StorageManager, Dataset


for name, url in [("spotify-tracks", 'https://www.kaggle.com/api/v1/datasets/download/saichaitanyareddyai/spotify-tracks-dataset-audio-features'), ('spotify-lyrics', 'https://www.kaggle.com/api/v1/datasets/download/evabot/spotify-lyrics-dataset')]:
    # https://app.clear.ml/datasets
    zip_file = StorageManager.download_file(
        remote_url=url
    )

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall() 
        csv_path = z.namelist()[0]
    
    # Create a dataset with ClearML`s Dataset class
    dataset = Dataset.create(
        dataset_project="HSE-GP5", dataset_name=name
    )    

    # add the example csv
    dataset.add_files(path=csv_path)

    # Upload dataset to ClearML server (customizable)
    dataset.upload()

    # commit dataset changes
    dataset.finalize()
    os.remove(csv_path)
    os.remove(zip_file)