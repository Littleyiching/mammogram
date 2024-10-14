
from dataprocess import device, Import_FullMammo, local_directory
from DLprocessing import train, test_model, Load_patchdata
import torch.optim as optim
from torch import nn
import pandas as pd
from mil import Attention
import os
# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
dir_path="{}/../pth".format(local_directory)
trainset, testset = Import_FullMammo()
train_loader, valid_loader, test_loader = Load_patchdata(trainset, testset, batch_size=1)
model = Attention()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5)

print(model)

train(model, train_loader, valid_loader, optimizer, 50)

