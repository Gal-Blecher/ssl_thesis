import yaml
from yaml.loader import SafeLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

setup = {'batch_size': 128,
         'dataset_name': 'cifar10',
         'experiment_name': 'vae_2',
         'lr': 1e-2,
         'n_epochs': 50
         }

def run_config():
    pass

if __name__ == '__main__':
    pass