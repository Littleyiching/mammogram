'''
Train on DDSM and Test on Mayo
'''
from dataprocess import device, Import_CropImg, local_directory
from DLprocessing import train_model, test_model, split_data, data_transformation_padding, MyDataset, plot_roc_curve
import torch.optim as optim
from torch import nn
import pandas as pd
from torchvision import datasets

import os
from torchvision import models
import torch
import re
from dinov2_model import DinoVisionTransformerClassifier

# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
dir_path="{}/../pth/combine".format(local_directory)
trainset, testset = Import_CropImg()
data_transforms = data_transformation_padding()
trainset, validset = split_data(trainset)
DDSM_train = MyDataset(trainset, transform=data_transforms['train'])
DDSM_valid = MyDataset(validset, transform=data_transforms['val'])
DDSM_test = MyDataset(testset, transform=data_transforms['val'])

# Create data loaders, use a batch size of '64', set shuffle to 'False' and workers to '0'
batch_size = 32
train_loader = torch.utils.data.DataLoader(DDSM_train, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(DDSM_valid, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(DDSM_test, batch_size=batch_size, shuffle=False, num_workers=2)

method = ['dino_g']
dino_g = DinoVisionTransformerClassifier("giant")

model_list = [dino_g]
num_classes=2
loss_module = nn.CrossEntropyLoss()
assert len(method) == len(model_list)
true_label_list = []
predict_prob_list = []
for i, model in enumerate(model_list):
    if re.search("ViT", method[i])!=None:
        # Get the number of input features to the final classification layer
        num_ftrs = model.heads.head.in_features
        # Replace the classification head
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif re.search("swin", method[i])!=None:
        num_ftrs = model.head.in_features
        # Replace the classification head
        model.head = nn.Linear(num_ftrs, num_classes)
    elif re.search("maxvit", method[i])!=None:
        num_ftrs=model.classifier[5].in_features
        model.classifier[5] = nn.Linear(num_ftrs, num_classes)
    elif re.search("dense", method[i])!=None:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif method[i] == 'efficientv2' or method[i] == 'mnas_05':
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif re.search("convnext", method[i])!=None:
        num_ftrs=model.classifier[2].in_features
        model.classifier[2]=nn.Linear(num_ftrs, 2)
    elif re.search("vgg", method[i])!=None:
        num_ftrs=model.classifier[6].in_features
        model.classifier[6]=nn.Linear(num_ftrs, 2)
    elif re.search("dino", method[i])!=None:
        print("dinov2!")
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"====={method}====")
    model.cuda()
    if re.search("dino", method[i])!=None:
        optimizer = optim.Adam(model.parameters(), lr=5e-6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=100)
    scheduler_name = scheduler.__class__.__name__
    id_name = f"{current_file_name}-{method[i]}"
    training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, id_name, 50)
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/{id_name}_best.pth", id_name, device)
    result = pd.DataFrame({'training loss': training_losses,
                        'validation loss': validation_losses,
                        'training accuracy': training_acc,
                        'validation accuracy': validation_acc})
    result.to_csv(f'{dir_path}/{id_name}.csv', index=False) 
    true_label_list.append(true_labels)
    predict_prob_list.append(predicted_probabilities)

plot_roc_curve(true_label_list, predict_prob_list, method, f"{current_file_name}-dino_g")

