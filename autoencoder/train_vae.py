from CNNVae import VariationalAutoencoder
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from config.vae import *
import torch
import shutil
import os

config_path = 'config/vae.py'
modelname = 'vae24'
log_path = os.path.join("autoencoder/runs",modelname)
checkpoint_path = os.path.join("autoencoder/model",modelname)
data_dir = 'autoencoder/dataset/'

assert not os.path.exists(checkpoint_path)


# Applying Transformation
train_transforms = transforms.Compose([
                        transforms.Resize((245, 245)),
                        transforms.Grayscale(),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: ((x * 255.0) / 4.0)), 
                        ])

test_transforms = transforms.Compose([transforms.Resize((245, 245)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: ((x * 255.0) / 4.0))
                        ])

train_data = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
# test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

m=len(train_data)
train_data, val_data = random_split(train_data, [round(m-m*0.15), round(m*0.15)])


# Data Loading
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=4,pin_memory=True)
validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=4,pin_memory=True)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

model = VariationalAutoencoder(latent_dims=LATENT_SPACE)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


model.Trainer(trainloader=trainloader,
                valloader=validloader,
                optim=optim,
                num_epochs = NUM_EPOCHS,
                log_path=log_path,
                checkpoint_path = checkpoint_path)


shutil.copy(config_path, os.path.join(checkpoint_path, os.path.basename(config_path)))