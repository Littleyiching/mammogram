
from dataprocess import device, Import_CropImg
from DLprocessing import Load_data, train_model, plot_learning_curves, plot_accuracy, test_model
import torch.optim as optim
from torch import nn

from torchvision import models
import pandas as pd
import os
import torch
import re

# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
trainset, testset = Import_CropImg()

num_classes = 2
loss_module = nn.CrossEntropyLoss()
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []
dir_path="/xtra/ho000199/temp"
method = ['convnext_tiny']
#convnext_s = models.convnext_small()
convnext_t = models.convnext_tiny()
#convnext_b = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
#vgg16 = models.vgg16()
#convnext_s.load_state_dict(torch.load("/xtra/ho000199/temp/pth/pretrained/convnext_small-0c510722.pth"))
convnext_t.load_state_dict(torch.load("/xtra/ho000199/temp/pth/pretrained/convnext_tiny-983f1562.pth"))
#vgg16.load_state_dict(torch.load("/xtra/ho000199/temp/pth/pretrained/vgg16-397923af.pth"))
model_list = [convnext_t]

assert len(method) == len(model_list)
train_loader, valid_loader, test_loader = Load_data(trainset, testset, aug='imgnet')
for i, model in enumerate(model_list):
    if re.search("ViT", method[i])!=None:
        # Get the number of input features to the final classification layer
        num_ftrs = model.heads.head.in_features
        # Replace the classification head
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif re.search("swin_v2_b", method[i])!=None:
        num_ftrs = model.head.in_features
        # Replace the classification head
        model.head = nn.Linear(num_ftrs, num_classes)
    elif re.search("maxvit", method[i])!=None:
        num_ftrs=model.classifier[5].in_features
        model.classifier[5] = nn.Linear(num_ftrs, num_classes)
    elif re.search("densenet121", method[i])!=None:
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
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    print(model)
    model.cuda()
    #optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=100)
    scheduler_name = scheduler.__class__.__name__
    
    training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'{method[i]}', 100)
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

