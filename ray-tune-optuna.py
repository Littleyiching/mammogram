import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split
import re

import numpy as np
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from filelock import FileLock
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.train import RunConfig
from ray.train import CheckpointConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
import optuna
from ray.tune.search.optuna import OptunaSearch
from torchvision import transforms

# please do not modify this!
seed = 89802024

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import models
class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dir = "/xtra/ho000199"
        image_path = data_dir + os.sep + self.data.image_path[index]
        path_lock = image_path+'.lock'
        with FileLock(path_lock):
          image = Image.open(image_path).convert('RGB')
        label = self.data.label[index] #I guess this is your class
        if self.transform:
            image = self.transform(image)
        return image, label
def data_augmentation(config):
    target_size = (256, 256)
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=config["degree"]),
            transforms.RandomApply([transforms.RandomResizedCrop(size=target_size, scale=(0.75, 1.25))], p=config["prob1"]),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=config["prob2"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
    }
    print("=========data_augmentation")
    return data_transforms

def Load_alldata(config):
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
    # Split the dataset into training and validation sets
    trainset, validset = train_test_split(trainset, test_size=0.2)
    trainset=trainset.reset_index()
    validset=validset.reset_index()

    data_transforms = data_augmentation(config)
    train_dataset = MyDataset(trainset, transform=data_transforms['train'])
    valid_dataset = MyDataset(validset, transform=data_transforms['val'])
    test_dataset = MyDataset(testset, transform=data_transforms['val'])
    return train_dataset, valid_dataset, test_dataset

def train_func(config):

    num_epoch=50
    batch_size=32
    model=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs=model.classifier[2].in_features
    model.classifier[2]=nn.Linear(num_ftrs, 2)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if config["onecycle"]:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)


    train_dataset, valid_dataset, _ = Load_alldata(config)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if config["onecycle"]:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epoch)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if config["onecycle"]:
                scheduler.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valid_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        if config["onecycle"] == False:
            scheduler.step(val_loss)
        
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")

def test_best_model(best_result):

    best_trained_model=models.convnext_tiny()
    num_ftrs=best_trained_model.classifier[2].in_features
    best_trained_model.classifier[2]=nn.Linear(num_ftrs, 2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    _, _, test_dataset = Load_alldata(best_result.config)

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
    "prob1": tune.uniform(0, 1),
    "prob2": tune.uniform(0, 1),
    "degree": tune.choice([30, 180]),
    "onecycle": tune.choice([True, False]),
    }
    # Define the directory where results will be saved

    async_hyperband = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10
    )   

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 20, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=async_hyperband,
            num_samples=num_samples,
            search_alg=OptunaSearch(
                metric="accuracy",
                mode="max"
            ),
        ),
        param_space=config,
        run_config=RunConfig(
        name="conv",
        checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        # *Best* checkpoints are determined by these params:
        checkpoint_score_attribute="accuracy",
        checkpoint_score_order="max",
        ),
        storage_path="/xtra/ho000199/temp/ray_result",
        #stop={"accuracy": 0.8},
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

main(num_samples=30, max_num_epochs=50, gpus_per_trial=1)
