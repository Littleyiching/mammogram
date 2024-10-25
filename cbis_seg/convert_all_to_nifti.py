import re
import os
from shutil import copyfile
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import numpy as np

task_num = '088'
task_name = 'CBIS_Nifti'
json_description = "mammogram"
json_modalities = [("zscore")]


task_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset088_Mammography' + task_num +'_' +task_name +'/'
#seg_prefix = '_mask.nii.gz'
#img_prefix = '_image.nii.gz'

json_name = task_name
json_tensorImageSize ='2D'
json_reference = ""
json_license = 'private'
json_release ="0.0"
json_labels = {"background": "0", "legion": "1"}
dir_path='/research/m323170/Projects/mammography'

# mkdirs if not already
if not os.path.exists(task_folder):
	os.makedirs(task_folder)

if not os.path.exists(task_folder + 'imagesTr/'):
	os.makedirs(task_folder + 'imagesTr/')
if not os.path.exists(task_folder + 'imagesTs/'):
	os.makedirs(task_folder + 'imagesTs/')
if not os.path.exists(task_folder + 'labelsTr/'):
	os.makedirs(task_folder + 'labelsTr/')
local_directory = os.getcwd()
dicom_info = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/dicom_info.csv')
mass_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_test_set.csv')
calc_train = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_train_set.csv')
calc_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_test_set.csv')
train = ['Mass-Training', 'Calc-Training', 'Mass-Test', 'Calc-Test']

train_list = [mass_train, calc_train, mass_test, calc_test]
dicom_cropped=dicom_info[dicom_info['SeriesDescription']=='cropped images']
dicom_full=dicom_info[dicom_info['SeriesDescription']=='full mammogram images']
dicom_mask=dicom_info[dicom_info['SeriesDescription']!='cropped images']
orig_filelist = []
orig_masklist = []
train_id = []

for idx in range(len(train)):
    SeriesUID_mask = train_list[idx]['ROI mask file path'].str.extract(re.escape(train[idx])+r'.*\/.*\/(.*)\/')
    SeriesUID = train_list[idx]['image file path'].str.extract(re.escape(train[idx])+r'.*\/.*\/(.*)\/')
    path_list = []
    assert len(SeriesUID_mask) == len(SeriesUID)
    for ipath in range(len(SeriesUID)):
        if SeriesUID[0][ipath] in dicom_info.SeriesInstanceUID.values:
            orig_full = f"{dir_path}/{dicom_info[dicom_info['SeriesInstanceUID'] == SeriesUID[0][ipath]].image_path.item()}"  
            if orig_full in orig_filelist:
                continue
            orig_filelist.append(orig_full)
        else:
            print('[train]full not exist')
            print(f"full:{SeriesUID[0][ipath]}")
            print(f"mask:{SeriesUID_mask[0][ipath]}")
            train_list[idx].drop([ipath], inplace=True)
        if SeriesUID_mask[0][ipath] in dicom_mask.SeriesInstanceUID.values:
            orig_mask = f"{dir_path}/{dicom_mask[dicom_mask['SeriesInstanceUID'] == SeriesUID_mask[0][ipath]].image_path.item()}"
            orig_masklist.append(orig_mask)
            train_id.append(SeriesUID[0][ipath])
        else:
            print('[train]mask not exist')
            print(f"full:{SeriesUID[0][ipath]}")
            print(f"mask:{SeriesUID_mask[0][ipath]}")
            train_list[idx].drop([ipath], inplace=True)
            orig_filelist.remove(orig_full)  
    dataset=train_list[idx]
print(f"trainset:{len(orig_filelist)}, {len(orig_masklist)}, {len(train_id)}")
import nibabel as nib

import os
import cv2
import nibabel as nib

class PNGToNiftiConverter:
    def __init__(self, clipLimit=2.0, tileGridSize=8):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def convert_pngs_to_nifti_with_clahe(self, input_folder, output_folder):
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Loop through each PNG file in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.png'):
                # Load the image in grayscale
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileGridSize, self.tileGridSize))
                clahe_image = clahe.apply(image)

                # Convert to NIfTI format
                image_np = np.array(clahe_image)  # No need for PIL conversion
                nifti_image = nib.Nifti1Image(image_np, affine=np.eye(4))  # Keep simple identity affine

                # Save as NIfTI file in the output folder
                nifti_filename = os.path.splitext(filename)[0] + '.nii.gz'
                output_path = os.path.join(output_folder, nifti_filename)
                nib.save(nifti_image, output_path)

                print(f"Converted {filename} to {nifti_filename} with CLAHE normalization.")

# Set your input and output directories
input_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset077_Mammography077_CBIS/imagesTr/'  # Replace with the path to your PNG folder
output_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset088_Mammography088_CBIS_Nifti/imagesTr/'  # Replace with the desired output folder

# Initialize and run the converter
converter = PNGToNiftiConverter(clipLimit=2, tileGridSize=32)
converter.convert_pngs_to_nifti_with_clahe(input_folder, output_folder)

n_train=len(orig_masklist)
test_id = []
json_dict = {}
json_dict['name'] = json_name
json_dict['description'] = json_description
json_dict['tensorImageSize'] = json_tensorImageSize
json_dict['reference'] = json_reference
json_dict['licence'] = json_license
json_dict['release'] = json_release
json_dict['channel_names'] = {str(i): json_modalities[i] for i in range(len(json_modalities))}
json_dict['labels'] = {str(i): json_labels[i] for i in json_labels.keys()}

json_dict['numTraining'] = n_train
json_dict['file_ending'] = ".nii.gz"
json_dict['numTest'] = 0

json_dict['training'] = [
        {'image': "./imagesTr/" +  train_id[idx] + '_0000.nii.gz', "label": "./labelsTr/" +  train_id[idx] + '.nii.gz'} for idx in range(len(train_id))]
json_dict['test'] = ["./imagesTs/" + test_id[ipath]+ '_0000.nii.gz' for ipath in range(len(test_id))]

save_json(json_dict, task_folder + "dataset.json")