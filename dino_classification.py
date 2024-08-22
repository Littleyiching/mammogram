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
from DLprocessing import train_and_save_metrics, plot_learning_curves, plot_accuracy, test_model
from dinov2_model import DinoVisionTransformerClassifier
from cvprocessing import Load_data_with_path

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

loss_module=nn.CrossEntropyLoss()
method = ["clahe", "none"]
for i in range(3):
    model = DinoVisionTransformerClassifier("base")
    model = model.to(device)
    print(model)
    train_loader, valid_loader, test_loader = Load_data_with_path(trainset, testset, m='aug')
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    scheduler_name = scheduler.__class__.__name__

    training_losses, validation_losses, training_acc, validation_acc = train_and_save_metrics(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'dino_{i}', 40)
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/dino_{i}_best.pth", scheduler_name, device)
    result = pd.DataFrame({'training loss': training_losses,
                        'validation loss': validation_losses,
                        'training accuracy': training_acc,
                        'validation accuracy': validation_acc})
    result.to_csv(f'dino_{i}.csv', index=False) 
    train_loss_list.append(training_losses)
    valid_loss_list.append(validation_losses)
    train_acc_list.append(training_acc)
    valid_acc_list.append(validation_acc)
    
train_avg = [sum(x) / len(x) for x in zip(*train_loss_list)]
valid_avg = [sum(x) / len(x) for x in zip(*valid_loss_list)]
trainacc_avg = [sum(x) / len(x) for x in zip(*train_acc_list)]
validacc_avg = [sum(x) / len(x) for x in zip(*valid_acc_list)]
print(f"train loss avg:{train_avg}, val loss avg:{valid_avg}, train acc avg:{trainacc_avg}, val acc avg:{validacc_avg}")
avg_result = pd.DataFrame({'training loss': train_avg,
                    'validation loss': valid_avg,
                    'training accuracy': trainacc_avg,
                    'validation accuracy': validacc_avg})
avg_result.to_csv(f'dino_avg.csv', index=False) 
