
from dataprocess import Import_CropImg, device
import torch
import os
from torchvision import models
from torch import nn
import re
from dinov2_model import DinoVisionTransformerClassifier
from cvprocessing import Load_CLAHE_data
# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
trainset, testset = Import_CropImg()
_, _, test_loader = Load_CLAHE_data(trainset, testset)

method = ["../ray_result/dino_clahe/train_func_6d5587ac_4_aug=False,cl=0.0500,grid=5,version=base_2024-08-20_15-43-34/checkpoint_000009/checkpoint.pt", "../ray_result/dino_clahe/train_func_4414b3f3_23_aug=False,cl=0.0500,grid=5,version=base_2024-08-20_17-22-24/checkpoint_000009/checkpoint.pt", "../ray_result/dino_clahe/train_func_a91b63c3_11_aug=False,cl=0.0500,grid=5,version=base_2024-08-20_16-22-15/checkpoint_000039/checkpoint.pt", '../ray_result/dino_clahe/train_func_f5054cd7_6_aug=False,cl=0.0100,grid=3,version=small_2024-08-20_15-52-18/checkpoint_000049/checkpoint.pt', '../ray_result/dino_clahe/train_func_56b184fd_12_aug=False,cl=0.0500,grid=5,version=small_2024-08-20_16-26-34/checkpoint_000009/checkpoint.pt','../ray_result/dino_clahe/train_func_1f67b2bc_1_aug=False,cl=2,grid=3,version=base_2024-08-20_15-43-18/checkpoint_000039/checkpoint.pt', '../ray_result/dino_clahe/train_func_75eda61b_9_aug=False,cl=0.0500,grid=3,version=small_2024-08-20_16-20-21/checkpoint_000009/checkpoint.pt', '../ray_result/dino_clahe/train_func_fce3b54c_16_aug=False,cl=0.0500,grid=5,version=base_2024-08-20_16-44-17/checkpoint_000009/checkpoint.pt', '../ray_result/dino_clahe/train_func_f3f39f80_19_aug=False,cl=0.0500,grid=5,version=base_2024-08-20_16-57-26/checkpoint_000039/checkpoint.pt']
def test_model(model, test_loader, path, name, device):

    true_labels = []

    # predicted probabilities have probability that a sample belongs to positive class (having pneumonia)
    predicted_probabilities = []
    
    model_state, optimizer_state = torch.load(path)
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
#method = ["transformer_models-convnext_s_best.pth"]
for i, path in enumerate(method):
    if re.search("convnext", path) != None:
        if re.search("convnext_s", path)!=None:
            print("convsmall")
            model=models.convnext_small()
        elif re.search("convnext_t", path)!=None:
            model=models.convnext_tiny()
        elif re.search("convnext_b", path)!=None:
            model=models.convnext_base()
        num_ftrs=model.classifier[2].in_features
        model.classifier[2]=nn.Linear(num_ftrs, 2)
    elif re.search("vgg", path)!=None:
        if re.search("vgg11", path)!=None:
            model=models.vgg11()
        elif re.search("vgg13", path)!=None:
            model=models.vgg13()
        elif re.search("vgg16", path)!=None or re.search("vgg", path):
            model=models.vgg16()
        else:
            model=models.vgg19()
        num_ftrs=model.classifier[6].in_features
        model.classifier[6]=nn.Linear(num_ftrs, 2)
    elif re.search("mnas", path)!=None:
        model=models.mnasnet0_5()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1]=nn.Linear(num_ftrs, 2)
    elif re.search("efficient", path)!=None:
        model=models.efficientnet_v2_s()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif re.search("dino", path)!=None and re.search("base", path)!=None:
        model = DinoVisionTransformerClassifier("base")
    elif re.search("dino", path)!=None and re.search("small", path)!=None:
        model = DinoVisionTransformerClassifier("small")
    else:
        if re.search("shuffle", path)!=None:
            model=models.shufflenet_v2_x0_5()
        elif re.search("wide", path)!=None:
            model=models.wide_resnet50_2()
        else:
            model=models.regnet_y_400mf()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
#    model.cuda()
    model.to(device)
    true_labels, predicted_probabilities = test_model(model, test_loader, path, path, device)
    #true_labels, predicted_probabilities = test_model(model, test_loader, path, path, device)

