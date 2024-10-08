import os
import random

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
import os
import torch
from torch import nn, optim
#import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
from dataprocess import device, local_directory, Import_CropImg
from DLprocessing import split_data
from dinov2_model import DinoVisionTransformerClassifier
import os

# please do not modify this!
seed = 89802024

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class Dataset_CLAHE(Dataset):
    def __init__(self, dataset, clipLimit=0.01, tileGridSize=8, transform=None):
        self.data = dataset
        self.transform = transform
        self.clipLimit=clipLimit
        self.tileGridSize=tileGridSize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dir = "{}/..".format(local_directory)
        image_path = data_dir + os.sep + self.data.image_path[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Step 4: Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileGridSize, self.tileGridSize))
        clahe_image = clahe.apply(image)

        # Step 5: Convert the NumPy array back to a PIL image (optional)
        image = Image.fromarray(clahe_image).convert('RGB')
        label = self.data.label[index] #I guess this is your class
        if self.transform:
            image = self.transform(image)
        return image, label
    
# These are settings for ensuring input images to DinoV2 are properly sized
class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        # Resize the image
        img = transforms.Resize(self.target_size)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = transforms.Pad((pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)

        return img
    
def data_transformation_aug():
    target_size = (224, 224)
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            ResizeAndPad(target_size, 14),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.AugMix(),
            transforms.ToTensor(),           # Convert images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
        ]),
        'val': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
        ]),
    }
    print("=========augmix")
    return data_transforms    
def data_transformation_imgnet():
    target_size = (224, 224)
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            ResizeAndPad(target_size, 14),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),           # Convert images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
        ]),
        'val': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
        ]),
    }
    print("=========rotate 180, flip")
    return data_transforms

def Load_alldata(config):
    trainset, testset = Import_CropImg()
    trainset, validset = split_data(trainset)

    if config["aug"]:
        data_transforms = data_transformation_aug()
    else:
        data_transforms = data_transformation_imgnet()
    train_dataset = Dataset_CLAHE(trainset, clipLimit=config["cl"], tileGridSize=config["grid"], transform=data_transforms['train'])
    valid_dataset = Dataset_CLAHE(validset, transform=data_transforms['val'])
    test_dataset = Dataset_CLAHE(testset, transform=data_transforms['val'])
    return train_dataset, valid_dataset, test_dataset


def train_func(config):

    num_epoch=50
    batch_size=16
    if config["version"] == "small":
        model = DinoVisionTransformerClassifier("small")
    elif config["version"] == "base":
        model = DinoVisionTransformerClassifier("base")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
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

    best_trained_model = DinoVisionTransformerClassifier("small")
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
    "version": tune.choice(["small", "base"]),
    "grid": tune.choice([3, 5, 8]),
    "cl": tune.choice([0.01, 0.05, 1, 2]),
    "aug": tune.choice([True, False]),
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
        name="dino_clahe",
        checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        # *Best* checkpoints are determined by these params:
        checkpoint_score_attribute="accuracy",
        checkpoint_score_order="max",
        ),
        storage_path="{}/../ray_result".format(local_directory),
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
