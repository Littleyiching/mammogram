import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import random

import re

import pandas as pd
import torch
local_directory = os.getcwd()
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

def Import_FullMammo():
    dir_path='{}/..'.format(local_directory)
    dicom_info = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/dicom_info.csv')
    mass_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_train_set.csv')
    mass_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_test_set.csv')
    calc_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_train_set.csv')
    calc_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_test_set.csv')
    pattern = ['Mass-Training', 'Mass-Test', 'Calc-Training', 'Calc-Test']
    data_list = [mass_train, mass_test, calc_train, calc_test]
    path_list = []
    for idx in range(len(pattern)):
        SeriesUID = data_list[idx]['image file path'].str.extract(re.escape(pattern[idx])+r'.*\/.*\/(.*)\/')
        path_list = []
        for ipath in range(len(SeriesUID)):
            path_list.append(dicom_info[dicom_info['SeriesInstanceUID'] == SeriesUID[0][ipath]].image_path.item())
        data_list[idx]['image_path']=path_list
        # find case number within each catogory
        dataset=data_list[idx]
        dataset['label'] = np.where(dataset['pathology']=='MALIGNANT', 1, 0)
    calc_train=calc_train.rename(columns={'breast density': 'breast_density'})
    calc_test=calc_test.rename(columns={'breast density': 'breast_density'})
    trainset=pd.concat([mass_train, calc_train], ignore_index=True)
    testset=pd.concat([mass_test, calc_test], ignore_index=True)
    print(len(trainset))
    print(len(testset))
    return trainset, testset

def Import_CropImg():
    dir_path='{}/..'.format(local_directory)
    dicom_info = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/dicom_info.csv')
    mass_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_train_set.csv')
    mass_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_test_set.csv')
    calc_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_train_set.csv')
    calc_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_test_set.csv')
    pattern = ['Mass-Training', 'Mass-Test', 'Calc-Training', 'Calc-Test']
    data_list = [mass_train, mass_test, calc_train, calc_test]
    dicom_cropped=dicom_info[dicom_info['SeriesDescription']=='cropped images']

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
    return trainset, testset

def Import_CropDDSM():
    dir_path='{}/..'.format(local_directory)
    dicom_info = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/dicom_info.csv')
    mass_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_train_set.csv')
    mass_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_test_set.csv')
    calc_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_train_set.csv')
    calc_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_test_set.csv')
    pattern = ['Mass-Training', 'Mass-Test', 'Calc-Training', 'Calc-Test']
    data_list = [mass_train, mass_test, calc_train, calc_test]
    dicom_cropped=dicom_info[dicom_info['SeriesDescription']=='cropped images']

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
    trainset=pd.concat([mass_train, calc_train, mass_test, calc_test], ignore_index=True)
    #testset=pd.concat([mass_test, calc_test], ignore_index=True)

    # same patient cases with benign findings
    print("Total Benign cases for training")
    print(len(trainset[(trainset['pathology']=='BENIGN') | (trainset['pathology']=='BENIGN_WITHOUT_CALLBACK')]))
    print("Total malignant cases for training")
    print(len(trainset[(trainset['pathology']=='MALIGNANT')]))
    return trainset

# Validation function to identify and plot false positives and false negatives
def save_metrics(preds, labels, image_path, idx, name):
    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []
    assert len(preds) == len(labels) == len(image_path)
    for i in range(len(labels)):
        if preds[i] == 1 and labels[i] == 0:
            false_positives.append(image_path[i])
        elif preds[i] == 0 and labels[i] == 1:
            false_negatives.append(image_path[i])
        elif preds[i] == 1 and labels[i] == 1:
            true_positives.append(image_path[i])
        else:
            true_negatives.append(image_path[i])
    data = {'false positive': false_positives,
                'false negative': false_negatives,
                'true positive': true_positives,
                'true negative': true_negatives}
    result = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    if idx == 0:
      result.to_csv(f'metrics-{name}.csv', index=False)
    else:
      result.to_csv(f'metrics-{name}.csv', mode='a', header=False, index=False)
