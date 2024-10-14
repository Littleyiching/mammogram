import os
from torch import nn

import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
from sklearn.model_selection import train_test_split
import re

import numpy as np
from torch.utils.data import Dataset
from DLprocessing import data_transformation_imgnet
from vgg import VGG, get_vgg_layers
from skorch.helper import SliceDataset
import torch
from PIL import Image
from torchvision import models
class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dir_path='/xtra/ho000199'
        data_dir = dir_path
        image_path = data_dir + os.sep + self.data.image_path[index]
        image = Image.open(image_path).convert('RGB')
        label = self.data.label[index] #I guess this is your class
        if self.transform:
            image = self.transform(image)
        return image, label
def Load_data():
    dir_path='/xtra/ho000199'
    dicom_info = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/dicom_info.csv')
    mass_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_train_set.csv')
    mass_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_test_set.csv')
    calc_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_train_set.csv')
    calc_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_test_set.csv')

    dicom_cropped=dicom_info[dicom_info['SeriesDescription']=='cropped images']
    pattern = ['Mass-Training', 'Mass-Test', 'Calc-Training', 'Calc-Test']

    data_list = [mass_train, mass_test, calc_train, calc_test]
    for idx in range(len(pattern)):
        SeriesUID = data_list[idx]['cropped image file path'].str.extract(re.escape(pattern[idx])+r'.*\/.*\/(.*)\/')
        path_list = []
        for ipath in range(len(SeriesUID)):
            if SeriesUID[0][ipath] in dicom_cropped.SeriesInstanceUID.values:
                path_list.append(dicom_cropped[dicom_cropped['SeriesInstanceUID'] == SeriesUID[0][ipath]].image_path.item())
            else:
                print('not exist')
                print(SeriesUID[0][ipath])
                data_list[idx].drop([ipath], inplace=True)
        data_list[idx]['image_path']=path_list
        # find case number within each catogory
        dataset=data_list[idx]
        dataset['label'] = np.where(dataset['pathology']=='MALIGNANT', 1, 0)
    calc_train=calc_train.rename(columns={'breast density': 'breast_density'})
    calc_test=calc_test.rename(columns={'breast density': 'breast_density'})
    trainset=pd.concat([mass_train, calc_train], ignore_index=True)
    testset=pd.concat([mass_test, calc_test], ignore_index=True)

    # same patient cases with benign findings
    print("Total Benign cases for training")
    print(len(trainset[(trainset['pathology']=='BENIGN') | (trainset['pathology']=='BENIGN_WITHOUT_CALLBACK')]))
    print("Total malignant cases for training")
    print(len(trainset[(trainset['pathology']=='MALIGNANT')]))
    train_subset=trainset[(trainset.subtlety == 2) | (trainset.subtlety ==1) | (trainset.breast_density == 3) | (trainset.breast_density == 4)]
    print("subset base on density and subtlety")
    print(len(train_subset))
    # Split the dataset into training and validation sets
    trainset, validset = train_test_split(train_subset, test_size=0.2)
    trainset=trainset.reset_index()
    validset=validset.reset_index()
    data_transforms = data_transformation_imgnet()
    train_dataset = MyDataset(trainset, transform=data_transforms['train'])
    valid_dataset = MyDataset(validset, transform=data_transforms['val'])
    test_dataset = MyDataset(testset, transform=data_transforms['val'])
    return train_dataset, valid_dataset, test_dataset

model = models.convnext_small()
num_ftrs=model.classifier[2].in_features
model.classifier[2]=nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("/xtra/ho000199/temp/transformer_models-convnext_s_epoch_50.pth"))
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Ensure the final layer is trainable
for param in model.classifier[2].parameters():
    param.requires_grad = True
train_dataset, _, _ = Load_data()

model = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss,
    batch_size=32,
    max_epochs=10,
    verbose=False
)

# Define the grid search parameters
param_dist = {
    'optimizer__lr': [0.00001, 0.0001, 0.001, 0.1],
    'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta,
                  optim.Adam, optim.Adamax, optim.NAdam]
}

d_loader_slice_X = SliceDataset(train_dataset, idx=0)
d_loader_slice_y = SliceDataset(train_dataset, idx=1)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=3, n_jobs=5)
grid_result = grid.fit(d_loader_slice_X, d_loader_slice_y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

