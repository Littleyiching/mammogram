
from dataprocess import device, Import_CropImg
from DLprocessing import Load_subset, train_model, test_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
trainset, testset = Import_CropImg()
train_loader, valid_loader, test_loader = Load_subset(trainset, testset)
dir_path="/xtra/ho000199/temp"
#vgg=models.vgg16()
#num_ftrs = vgg.classifier[6].in_features
#vgg.classifier[6] = nn.Linear(num_ftrs, 2)
#vgg.load_state_dict(torch.load(f"{dir_path}/pth/vgg_feature-4096_best.pth", map_location=torch.device('cpu')))
convnext_t = models.convnext_tiny()
num_ftrs=convnext_t.classifier[2].in_features
convnext_t.classifier[2]=nn.Linear(num_ftrs, 2)
convnext_t.load_state_dict(torch.load(f"{dir_path}/convnext_t_best.pth", map_location=torch.device('cpu')))
loss_module = nn.CrossEntropyLoss()
method = ['convnext_t']
model_list = [convnext_t]

label_list = []
pred_list = []
for loop in range(2):
    for i, model in enumerate(model_list):
        if loop == 0:
            print("freeze layers")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            if method[i] == 'vgg':
                # Ensure the final layer is trainable
                for param in model.classifier[6].parameters():
                    param.requires_grad = True
            else:
                for param in model.classifier[2].parameters():
                    param.requires_grad = True
        print(model)
        model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)    
        #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)
        scheduler_name = scheduler.__class__.__name__
        training_losses, validation_losses, training_acc, validation_acc = train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, f'{method[i]}_{scheduler_name}-{loop}', 50)
        true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/{method[i]}_{scheduler_name}-{loop}_best.pth", method[i], device)
