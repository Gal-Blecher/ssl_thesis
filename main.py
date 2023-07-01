from config import setup
import torch
import datasets
import train
import os
import expert_model

torch.manual_seed(42)

if __name__ == '__main__':
    path = './models/' + setup['experiment_name']
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = datasets.get_dataset()
    model = expert_model.VAE()
    train.train_vae(model, dataset, path)
    temp = 0
