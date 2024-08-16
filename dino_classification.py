# A simple notebook demonstrating how to fine-tune a DinoV2 classifier on your own images/labels

# Most of the core code was originally published in an excellent tutorial by Kili Technology, here:
#  https://colab.research.google.com/github/kili-technology/kili-python-sdk/blob/main/recipes/finetuning_dinov2.ipynb

# November 11th, 2023 by Lance Legel (lance@3co.ai) from 3co, Inc. (https://3co.ai)

import os
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn, optim
#import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
from torchvision import transforms
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from dataprocess import device, Import_CropImg
from DLprocessing import Load_data, train_model, plot_learning_curves, plot_accuracy, test_model
import torch.optim as optim
from torch import nn

import os
import torch
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
dir_path="/xtra/ho000199/temp"
local_directory = os.getcwd()

# Define a new classifier layer that contains a few linear layers with a ReLU to make predictions positive
class DinoVisionTransformerClassifier(nn.Module):

    def __init__(self, model_size="small"):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.model_size = model_size

        # loading a model with registers
        n_register_tokens = 4

        if model_size == "small":
            model = vit_small(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 384
            self.number_of_heads = 6

        elif model_size == "base":
            model = vit_base(patch_size=14,
                             img_size=526,
                             init_values=1.0,
                             num_register_tokens=n_register_tokens,
                             block_chunks=0)
            self.embedding_size = 768
            self.number_of_heads = 12

        elif model_size == "large":
            model = vit_large(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 1024
            self.number_of_heads = 16

        elif model_size == "giant":
            model = vit_giant2(patch_size=14,
                               img_size=526,
                               init_values=1.0,
                               num_register_tokens=n_register_tokens,
                               block_chunks=0)
            self.embedding_size = 1536
            self.number_of_heads = 24

        # Download pre-trained weights and place locally as-needed:
        # - small: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
        # - base:  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
        # - large: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
        # - giant: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth
        model.load_state_dict(torch.load(Path('dinov2_vits14_reg4_pretrain.pth'.format(local_directory))))

        self.transformer = deepcopy(model)

        self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

method=['dinov2_e6', 'dinov2_e4']
model = DinoVisionTransformerClassifier("small")
train_loader, valid_loader, test_loader = Load_data(trainset, testset, m='padding')

model = model.to(device)
# change the binary cross-entropy loss below to a different loss if using more than 2 classes
# https://pytorch.org/docs/stable/nn.html#loss-functions

loss_module=nn.CrossEntropyLoss()
for i, name in enumerate(method):
    if i == 0:
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    scheduler_name = scheduler.__class__.__name__

    training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'{name}', 100)
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/{name}_best.pth", scheduler_name, device)
    result = pd.DataFrame({'training loss': training_losses,
                        'validation loss': validation_losses,
                        'training accuracy': training_acc,
                        'validation accuracy': validation_acc})
    result.to_csv(f'{name}.csv', index=False) 
    train_loss_list.append(training_losses)
    valid_loss_list.append(validation_losses)
    train_acc_list.append(training_acc)
    valid_acc_list.append(validation_acc)
    
plot_learning_curves(train_loss_list, valid_loss_list, method, current_file_name)
plot_accuracy(train_acc_list, valid_acc_list, method, current_file_name)