from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from dataprocess import local_directory
from DLprocessing import split_data, data_transformation_imgnet, data_transformation_padding, data_augmentation, MyDataset
import cv2

dir_path="{}/..".format(local_directory)
   
class Dataset_CLAHE(Dataset):
    def __init__(self, dataset, clipLimit=0.05, tileGridSize=5, transform=None):
        self.data = dataset
        self.transform = transform
        self.clipLimit=clipLimit
        self.tileGridSize=tileGridSize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dir = dir_path
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
    
    
class Mydataset_with_path(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dir = dir_path
        image_path = data_dir + os.sep + self.data.image_path[index]
        image = Image.open(image_path).convert('RGB')
        label = self.data.label[index] #I guess this is your class
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

def Load_CLAHE_data(trainset, testset, m='imgnet'):
    batch_size=16
    trainset, validset = split_data(trainset)
    
    if m == 'imgnet':
        data_transforms = data_transformation_imgnet()
    elif m == 'padding':
        data_transforms = data_transformation_padding()
    else:
        data_transforms = data_augmentation()

    train_dataset = Dataset_CLAHE(trainset, transform=data_transforms['train'])
    valid_dataset = Dataset_CLAHE(validset, transform=data_transforms['val'])
    test_dataset = Dataset_CLAHE(testset, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader

def Load_data_with_path(trainset, testset, test_size=0.2, m='imgnet'):
    batch_size=32
    trainset, validset = split_data(trainset, test_size=test_size)
    
    if m == 'imgnet':
        data_transforms = data_transformation_imgnet()
    elif m == 'padding':
        data_transforms = data_transformation_padding()
    else:
        data_transforms = data_augmentation()

    train_dataset = Mydataset_with_path(trainset, transform=data_transforms['train'])
    valid_dataset = Mydataset_with_path(validset, transform=data_transforms['val'])
    test_dataset = MyDataset(testset, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader
