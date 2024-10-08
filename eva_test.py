
from dataprocess import Import_CropImg, device, local_directory
from DLprocessing import Load_data, test_model, plot_roc_curve
import os
from torchvision import models
from torch import nn
import re
#from test import all_checkpoints

# Get the current file name
current_file_name = os.path.basename(__file__)
# Split the file name and extension
current_file_name, _ = os.path.splitext(current_file_name)
trainset, testset = Import_CropImg()
_, _, test_loader = Load_data(trainset, testset)
dir_path="{}/../pth".format(local_directory)

method = ["convt_none_2_best.pth", "convt_none_3_best.pth", "convt_none_4_best.pth", "vgg_best.pth"]

#method = ["transformer_models-convnext_s_best.pth"]
for i, path in enumerate(method):
    if re.search("conv", path) != None:
        if re.search("convnext_s", path)!=None:
            model=models.convnext_small()
        elif re.search("convt", path)!=None:
            print("convnext_t")
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
    true_labels, predicted_probabilities = test_model(model, test_loader, f"{dir_path}/{path}", path, device)
    plot_roc_curve(true_labels, predicted_probabilities, {path})

