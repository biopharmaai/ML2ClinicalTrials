import torch, os, sys

torch.manual_seed(0) 
from trialbench.dataset import *

from tqdm import tqdm
import requests
import warnings
warnings.filterwarnings("ignore")

def download_all_data(save_path):
    '''
    Download all datasets from zenodo.org
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = "https://zenodo.org/records/14975339/files/all_task.zip?download=1"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = file.write(chunk)
                bar.update(size)
    
    print(f"All the datasets are downloaded to: {save_path}")

def load_data(task, phase, data_format='df'):
    '''
    Load data from the dataset
    task: str, name of the dattaskaset
    phase: str, phase of the clinical trial (e.g., "Phase 1", "Phase 2", "Phase 3", "Phase 4")
    data_format: str, 'dl' for dataloader.Dataloader, 'df' for pd.Dataframe
    '''
    if data_format == 'dl':
        return get_dataloader(task, phase)
    elif data_format == 'df':
        train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = get_dataloader(task, phase)
        train_df, valid_df, test_df = data_loader_to_frame(train_loader, task), data_loader_to_frame(valid_loader, task), data_loader_to_frame(test_loader, task)
        return train_df, valid_df, test_df, num_classes, tabular_input_dim
    else:
        raise ValueError("data_format should be either 'dl' or 'df'")

def load_model():
    pass

if __name__ == "__main__":
    task = 'dose_cls'
    phase = 'all'

    # from trialbench.function import load_data
    train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = load_data(task, phase, data_format='dl')
    label = train_loader.dataset.label_lst
    
