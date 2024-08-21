
from dataprocess import device, Import_CropImg, local_directory
from DLprocessing import train_model, test_model, Load_data
import torch.optim as optim
from torch import nn
import pandas as pd

import os
from torchvision import models
# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
dir_path="{}/../pth".format(local_directory)
trainset, testset = Import_CropImg()
train_loader, valid_loader, test_loader = Load_data(trainset, testset)
model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

for i in range(1):
    num_ftrs=model.classifier[2].in_features
    model.classifier[2]=nn.Linear(num_ftrs, 2)
    model.to(device)
    print(model)

    loss_module = nn.CrossEntropyLoss()

    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    scheduler_name = scheduler.__class__.__name__
    training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'{current_file_name}-{scheduler_name}', 50)

    result = pd.DataFrame({'training loss': training_losses,
                            'validation loss': validation_losses,
                            'training accuracy': training_acc,
                            'validation accuracy': validation_acc})
    result.to_csv(f'{current_file_name}-{scheduler_name}.csv', index=False)
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/{current_file_name}-{scheduler_name}_best.pth", scheduler_name, device)

