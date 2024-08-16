
from dataprocess import device, Import_FullMammo
from DLprocessing import Load_data, train_model, test_model
import torch.optim as optim
from torch import nn

from torchvision import models
import pandas as pd
import os
import torch
from monai.networks.nets import TorchVisionFCModel
# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
dir_path="/xtra/ho000199/temp"
trainset, testset = Import_FullMammo()
train_loader, valid_loader, test_loader = Load_data(trainset, testset, aug='imgnet')

model = TorchVisionFCModel(
            "inception_v3", num_classes=4, pretrained=True, use_conv=False, pool=None, bias=True
        )
print(model)
for i in range(1):
    model.load_state_dict(torch.load("/xtra/ho000199/breast_density_classification/models/model.pt"))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    model.to(device)

    loss_module = nn.CrossEntropyLoss()

    if i != 0:
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)    
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)
    scheduler_name = scheduler.__class__.__name__
#    training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'{current_file_name}-{scheduler_name}', 50)

#    result = pd.DataFrame({'training loss': training_losses,
#                            'validation loss': validation_losses,
#                            'training accuracy': training_acc,
#                            'validation accuracy': validation_acc})
#    result.to_csv(f'{current_file_name}-{scheduler_name}.csv', index=False)
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/pretrain_model-ReduceLROnPlateau_epoch_50.pth", scheduler_name, device)

