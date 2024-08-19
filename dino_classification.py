# A simple notebook demonstrating how to fine-tune a DinoV2 classifier on your own images/labels

# Most of the core code was originally published in an excellent tutorial by Kili Technology, here:
#  https://colab.research.google.com/github/kili-technology/kili-python-sdk/blob/main/recipes/finetuning_dinov2.ipynb

# November 11th, 2023 by Lance Legel (lance@3co.ai) from 3co, Inc. (https://3co.ai)

import os
import torch
from torch import nn, optim
#import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
from dataprocess import device, Import_CropImg, local_directory
from DLprocessing import Load_data, train_model, plot_learning_curves, plot_accuracy, test_model
from dinov2_model import DinoVisionTransformerClassifier

import os
import pandas as pd


# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
trainset, testset = Import_CropImg()

num_classes = 2
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []
local_directory = os.getcwd()
dir_path="{}/../pth".format(local_directory)

# Define a new classifier layer that contains a few linear layers with a ReLU to make predictions positive

method=['dino_lc', 'dino_reg']
dino_reg = DinoVisionTransformerClassifier("small")
dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
in_features = dinov2_vits14_lc.linear_head.in_features
dinov2_vits14_lc.linear_head = nn.Linear(in_features, 2)
train_loader, valid_loader, test_loader = Load_data(trainset, testset, m='padding')
model_list = [dinov2_vits14_lc, dino_reg]
# change the binary cross-entropy loss below to a different loss if using more than 2 classes
# https://pytorch.org/docs/stable/nn.html#loss-functions

loss_module=nn.CrossEntropyLoss()
for i, model in enumerate(model_list):
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    scheduler_name = scheduler.__class__.__name__

    training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'{method[i]}', 1)
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/{method[i]}_best.pth", scheduler_name, device)
    result = pd.DataFrame({'training loss': training_losses,
                        'validation loss': validation_losses,
                        'training accuracy': training_acc,
                        'validation accuracy': validation_acc})
    result.to_csv(f'{method[i]}.csv', index=False) 
    train_loss_list.append(training_losses)
    valid_loss_list.append(validation_losses)
    train_acc_list.append(training_acc)
    valid_acc_list.append(validation_acc)
    
plot_learning_curves(train_loss_list, valid_loss_list, method, current_file_name)
plot_accuracy(train_acc_list, valid_acc_list, method, current_file_name)