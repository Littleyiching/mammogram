from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
from sklearn.metrics import roc_curve, auc
from dinov2_model import ResizeAndPad
from dataprocess import local_directory, save_metrics

dir_path="{}/..".format(local_directory)
pth_path="{}/../pth/combine".format(local_directory)
class PatchDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data.label)

    def __getitem__(self, index):
        fixed_width = 64*6
        fixed_height = 64*8
        data_dir = dir_path
        image_path = data_dir + os.sep + self.data.image_path[index]
        image = Image.open(image_path)
        image = image.resize((fixed_width, fixed_height))
        # Define patch size
        patch_size = 64
        stride = patch_size // 2  # 50% overlap
        # Iterate and slice the image
        width, height = image.size
        patches = []
        '''
        for j in range(0, height, patch_size):
            for i in range(0, width, patch_size):
                box = (i, j, i + patch_size, j + patch_size)
                patch = image.crop(box)
                if self.transform:
                  patch = self.transform(patch)
                patches.append(patch)
        '''
        # Extract patches with overlap
        for j in range(0, height - patch_size + 1, stride):
            for i in range(0, width - patch_size + 1, stride):
                box = (i, j, i + patch_size, j + patch_size)
                patch = image.crop(box)
                if self.transform:
                    patch = self.transform(patch)
                patches.append(patch)
        bag = torch.stack(patches)
        label = self.data.label[index]
        return bag, label
    
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in subdirectories by class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Create a list of (image_path, label) tuples
        self.image_paths = []
        self.labels = []
        
        # Loop through each class subdirectory
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class CustomPatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in subdirectories by class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Create a list of (image_path, label) tuples
        self.image_paths = []
        self.labels = []
        
        # Loop through each class subdirectory
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        fixed_width = 28*30
        fixed_height = 28*40
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = image.resize((fixed_width, fixed_height))
        # Define patch size
        patch_size = 28
        # Iterate and slice the image
        width, height = image.size
        patches = []
        for j in range(0, height, patch_size):
            for i in range(0, width, patch_size):
                box = (i, j, i + patch_size, j + patch_size)
                patch = image.crop(box)
                if self.transform:
                  patch = self.transform(patch)
                patches.append(patch)
        bag = torch.stack(patches)
        label = self.labels[idx]
        
        return bag, label
class MyDataset(Dataset):
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
        return image, label

def data_transformation_imgnet():
    target_size = (299, 299)
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(target_size),  # Resize images to target size
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

def data_transformation_padding():
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
            ResizeAndPad(target_size, 14),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])  # Normalize with mean and standard deviation
        ]),
    }
    print("=========add padding for dinov2")
    return data_transforms

def data_transformation():
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2], std=[0.25])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2], std=[0.25])
        ]),
    }
    print("=========not imgnet")
    return data_transforms

def data_augmentation():
    target_size = (224, 224)
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.AugMix(),
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
    print("=========Apply autmix, rotate, flip")
    return data_transforms

def split_data(trainset):
    # Split the dataset into training and validation sets
    trainset, validset = train_test_split(trainset, test_size=0.2)
    trainset=trainset.reset_index()
    validset=validset.reset_index()
    return trainset, validset

def Load_data(trainset, testset, m='imgnet'):
    batch_size=32
    trainset, validset = split_data(trainset)
    
    if m == 'imgnet':
        data_transforms = data_transformation_imgnet()
    elif m == 'padding':
        data_transforms = data_transformation_padding()
    else:
        data_transforms = data_augmentation()

    train_dataset = MyDataset(trainset, transform=data_transforms['train'])
    valid_dataset = MyDataset(validset, transform=data_transforms['val'])
    test_dataset = MyDataset(testset, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader

def Load_patchdata(trainset, testset, batch_size=1):
    trainset, validset = split_data(trainset)
    
    data_transforms = data_transformation()

    train_dataset = PatchDataset(trainset, transform=data_transforms['train'])
    valid_dataset = PatchDataset(validset, transform=data_transforms['val'])
    test_dataset = PatchDataset(testset, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, valid_loader, test_loader

def Load_subset(trainset, testset):
    batch_size=64
    data_transforms = data_transformation_imgnet()
    trainset=trainset[(trainset.subtlety == 2) | (trainset.subtlety ==1) | (trainset.breast_density == 3) | (trainset.breast_density == 4)]
    print("subsets based on subtlety and breast density")
    trainset, validset = split_data(trainset)
    print(len(trainset))
    trainset=trainset.reset_index()
    validset=validset.reset_index()

    train_dataset = MyDataset(trainset, transform=data_transforms['train'])
    valid_dataset = MyDataset(validset, transform=data_transforms['val'])
    test_dataset = MyDataset(testset, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader

def train_model(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, path='model', epochs=20):

    training_losses = []
    validation_losses = []
    training_acc = []
    validation_acc = []
    best_acc = 0
    scheduler_name = scheduler.__class__.__name__

    for epoch in range(epochs):

        # Set model to train mode
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{epochs}", leave=False, unit='mini-batch')

        # Batch loop
        total = 0
        correct = 0
        for inputs, labels in progress_bar:

            # Move input data to device (only strictly necessary if we use GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            #inputs, labels = inputs.cuda(), labels.cuda()

            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()

            # Run the model on the input data and compute the outputs
            if model.__class__.__name__ == 'Inception3':
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            # Calculate the loss
            loss = loss_module(outputs, labels)

            # Perform backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()
            if scheduler_name == 'OneCycleLR':
                scheduler.step()

            # Calculate the loss for current iteration
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Set model to eval mode for validation
        model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():

            for inputs, labels in valid_loader:

                # For validation batches, calculate the output, and loss in a similar way
                #inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_module(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        valid_loss = running_loss / len(valid_loader)
        valid_acc = correct / total
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(valid_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f},  Valid Loss: {valid_loss:.4f}, Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.8f}')

        # Save the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'{pth_path}/{path}_best.pth')
            print(f'Best Epoch {epoch+1}')

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        validation_acc.append(valid_acc)
        training_acc.append(train_acc)
    # Save the last state of training process
    torch.save(model.state_dict(), f'{pth_path}/{path}_epoch_{epoch + 1}.pth')

    return training_losses, validation_losses, training_acc, validation_acc

def train(model, train_loader, valid_loader, optimizer, num_epoch):
    accumulation_steps = 16
    for epoch in range(1, num_epoch):
        model.train()
        train_loss = 0.
        train_error = 0.
        correct = 0
        total = 0

        for batch_idx, (bag, bag_label) in enumerate(train_loader):
            bag, bag_label = bag.cuda(), bag_label.cuda()
            loss, _ = model.calculate_objective(bag, bag_label)
            train_loss += loss.item()
            error, Y_hat = model.calculate_classification_error(bag, bag_label)
            train_error += error
            # Count correct predictions
            correct += Y_hat.eq(bag_label).sum().item()
            total += bag_label.size(0)
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                # step
                optimizer.step()
                # reset gradients
                optimizer.zero_grad()

        # calculate loss and error for epoch
        train_loss /= len(train_loader)
        train_error /= len(train_loader)
        train_acc = 100. * correct / total
        # Validation
        model.eval()
        total = 0
        correct = 0
        valid_loss = 0.
        valid_error = 0.
        with torch.no_grad():

            for batch_idx, (bag, bag_label) in enumerate(valid_loader):
                bag, bag_label = bag.cuda(), bag_label.cuda()

                # calculate loss and metrics
                loss, _ = model.calculate_objective(bag, bag_label)
                valid_loss += loss.item()
                error, Y_hat = model.calculate_classification_error(bag, bag_label)
                valid_error += error
                # Count correct predictions
                correct += Y_hat.eq(bag_label).sum().item()
                total += bag_label.size(0)

        # calculate loss and error for epoch
        valid_loss /= len(valid_loader)
        valid_error /= len(valid_loader)
        valid_acc = 100. * correct / total
        print(f'Epoch {epoch}/{num_epoch}, Train Loss: {train_loss:.4f},  Valid Loss: {valid_loss:.4f}, Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}')

def train_and_save_metrics(model, loss_module, optimizer, scheduler, train_loader, valid_loader, device, path='model', epochs=20):

    training_losses = []
    validation_losses = []
    training_acc = []
    validation_acc = []
    best_acc = 0

    for epoch in range(epochs):

        # Set model to train mode
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{epochs}", leave=False, unit='mini-batch')

        # Batch loop
        total = 0
        correct = 0
        total_pred = []
        total_labels= []
        imagepath_list = []
        for inputs, labels, image_path in progress_bar:

            # Move input data to device (only strictly necessary if we use GPU)
            #inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = inputs.to(device), labels.to(device)

            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()

            # Run the model on the input data and compute the outputs
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_module(outputs, labels)

            # Perform backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Calculate the loss for current iteration
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_pred.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            imagepath_list.extend(list(image_path))

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        if epoch > 30:
            print("====start saving training")
            idx = epoch -30
            save_metrics(total_pred, total_labels, imagepath_list, idx, f'{path}_Train')

        model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        total_pred = []
        total_labels= []
        imagepath_list = []
        with torch.no_grad():

            for inputs, labels, image_path in valid_loader:

                # For validation batches, calculate the output, and loss in a similar way
                inputs, labels = inputs.to(device), labels.to(device)
                #inputs, labels = input.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = loss_module(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                total_pred.extend(preds.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())
                imagepath_list.extend(list(image_path))
        valid_loss = running_loss / len(valid_loader)
        valid_acc = correct / total
        scheduler.step(valid_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f},  Valid Loss: {valid_loss:.4f}, Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.8f}')
        if epoch >30: 
            print("====start saving validation")
            idx = epoch -30
            save_metrics(total_pred, total_labels, imagepath_list, idx, f'{path}_Val')

        # Save the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'{pth_path}/{path}_best.pth')
            print(f'Best Epoch {epoch+1}')

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        validation_acc.append(valid_acc)
        training_acc.append(train_acc)

    torch.save(model.state_dict(), f'{pth_path}/{path}_epoch_{epoch + 1}.pth')

    return training_losses, validation_losses, training_acc, validation_acc


def plot_learning_curves(training_losses, validation_losses, method, name):

    plt.figure(figsize=(8, 6))
    for i in range(len(method)):
        # Plot epoch wise training and validation losses (both in the same plot)
        assert len(training_losses[i]) == len(validation_losses[i])

        # YOUR CODE HERE
        epochs = range(1, len(training_losses[i]) + 1)

        plt.plot(epochs, training_losses[i], label=f'{method[i]} Training')
        plt.plot(epochs, validation_losses[i], label=f'{method[i]} Validation')

    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #plt.show()
    plt.savefig(f'{name}-loss.png')

def plot_accuracy(training_acc, validation_acc, method, name):

    plt.figure(figsize=(8, 6))
    for i in range(len(method)):
        # Plot epoch wise training and validation losses (both in the same plot)
        assert len(training_acc[i]) == len(validation_acc[i])

        # YOUR CODE HERE
        epochs = range(1, len(validation_acc[i]) + 1)

        plt.plot(epochs, training_acc[i], label=f'{method[i]} Training')
        plt.plot(epochs, validation_acc[i], label=f'{method[i]} Validation')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{name}-accuracy.png')

def test_model(model, test_loader, path, name, device):

    true_labels = []

    # predicted probabilities have probability that a sample belongs to positive class (having pneumonia)
    predicted_probabilities = []
    
    model_state = torch.load(path)
    model.load_state_dict(model_state)
    #model.load_state_dict(torch.load(path))
    print(path)
    # Set model to eval mode
    model.eval()
    total=0
    correct = 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            #inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # Get the predicted class indices
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())

            # Compute softmax probabilities for each class and select the probability of the positive class
            probs = torch.softmax(outputs, dim=1)[:, 1]

            predicted_probabilities.extend(probs.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"{name}:Best trial test set accuracy: {(correct / total)}")

    return true_labels, predicted_probabilities

def plot_roc_curve(labels, probs, method, name):

    plt.figure(figsize=(8, 6))
    for i in range(len(method)):
        # Compute and plot the ROC curve and specify the AUC value in the legend (or within the plot somewhere).
        fpr, tpr, thresholds = roc_curve(labels[i], probs[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{method[i]} (AUC = {roc_auc:.2f})')
    #plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{pth_path}/{name}-roc.png')

def load(save_path, model):
    pretraind_dict = torch.load(save_path)
    model_dict =  model.state_dict()
    # Obtain the weights from pretrained_dict in the model_dict
    state_dict = {k:v for k,v in pretraind_dict.items() if ((k in model_dict.keys()) and (v.size()==model_dict[k].size()))}
    #state_dict = {k:v for k,v in pretraind_dict.items() if (k in model_dict.keys())}
    # only Update the weights from state_dict to model_dict
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
def load_weight_from_test_to_model(path, model, test_model, use_strict=False):

    if use_strict:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(path)
        new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        load(path, model)
    
    for key, para in model.state_dict().items():
        print(key)
        if key in test_model.state_dict().keys():
            print(torch.equal(para, test_model.state_dict()[key]))

def simple_train(model, optimizer, loss_module, train_loader, device, path, num_epoch=30):
    print('====Training start====')
    for epoch in range(num_epoch):

        # Set model to train mode
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epoch}", leave=False, unit='mini-batch')

        # Batch loop
        total = 0
        correct = 0
        for inputs, labels in progress_bar:

            # Move input data to device (only strictly necessary if we use GPU)
            #inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(), labels.cuda()

            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()

            # Run the model on the input data and compute the outputs
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_module(outputs, labels)

            # Perform backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()
            #scheduler.step()

            # Calculate the loss for current iteration
            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        print('Epoch [{}/{}], Loss: {:.4f}, ACC: {:.4f}'.format(epoch+1, num_epoch, train_loss, train_acc))
    # Save the last state of training process
    torch.save(model.state_dict(), f"{path}.pth")

def plot_decoderimg(model, test_loader, device, name):
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon = model(data)
            break

    plt.figure(dpi=250)
    fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    for i in range(7):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
    plt.savefig(f'{name}-decode.png')

