import re
import os
from shutil import copyfile
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import numpy as np

task_num = '033'
task_name = 'CBIS'
json_description = "mammogram"
json_modalities = [("zscore")]


task_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset033_Mammography' + task_num +'_' +task_name +'/'

dir_path='/research/m323170/Projects/mammography'

dicom_info = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/dicom_info.csv')
mass_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/mass_case_description_test_set.csv')
calc_test = pd.read_csv(f'{dir_path}/CBIS-DDSM/csv/calc_case_description_test_set.csv')

test = ['Mass-Test', 'Calc-Test']
test_list = [mass_test, calc_test]
dicom_cropped=dicom_info[dicom_info['SeriesDescription']=='cropped images']
dicom_full=dicom_info[dicom_info['SeriesDescription']=='full mammogram images']
dicom_mask=dicom_info[dicom_info['SeriesDescription']!='cropped images']
orig_filelist = []
orig_masklist = []
test_id = []

for idx in range(len(test)):
    SeriesUID_mask = test_list[idx]['ROI mask file path'].str.extract(re.escape(test[idx])+r'.*\/.*\/(.*)\/')
    SeriesUID = test_list[idx]['image file path'].str.extract(re.escape(test[idx])+r'.*\/.*\/(.*)\/')
    path_list = []
    assert len(SeriesUID_mask) == len(SeriesUID)
    for ipath in range(len(SeriesUID)):
        if SeriesUID[0][ipath] in dicom_info.SeriesInstanceUID.values:
            orig_full = f"{dir_path}/{dicom_info[dicom_info['SeriesInstanceUID'] == SeriesUID[0][ipath]].image_path.item()}"  
            if orig_full in orig_filelist:
                continue
            orig_filelist.append(orig_full)
        else:
            print('[test]full not exist')
            print(f"full:{SeriesUID[0][ipath]}")
            print(f"mask:{SeriesUID_mask[0][ipath]}")
            test_list[idx].drop([ipath], inplace=True)
        if SeriesUID_mask[0][ipath] in dicom_mask.SeriesInstanceUID.values:
            orig_mask = f"{dir_path}/{dicom_mask[dicom_mask['SeriesInstanceUID'] == SeriesUID_mask[0][ipath]].image_path.item()}"
            orig_masklist.append(orig_mask)
            test_id.append(SeriesUID[0][ipath])
        else:
            print('[test]mask not exist')
            print(f"full:{SeriesUID[0][ipath]}")
            print(f"mask:{SeriesUID_mask[0][ipath]}")
            test_list[idx].drop([ipath], inplace=True)
            orig_filelist.remove(orig_full)  
    dataset=test_list[idx]
print(f"testset:{len(orig_filelist)}, {len(orig_masklist)}, {len(test_id)}")
assert len(orig_masklist) == len(orig_filelist)

def resize_image(full, mask, des_full, des_mask):
    # Load the two images
    image1 = Image.open(full)
    image2 = Image.open(mask)

    # Get the size (width, height) of both images
    size1 = image1.size  # (width, height) of image 1
    size2 = image2.size  # (width, height) of image 2

    # Determine the smaller size
    new_size = (min(size1[0], size2[0]), min(size1[1], size2[1]))

    # Resize both images to the smaller size
    image1_resized = image1.resize(new_size)
    image2_resized = image2.resize(new_size)
    # Convert the image to a NumPy array
    mask_array = np.array(image2_resized)

    # Convert all non-zero values to 1, keeping 0 as is
    binary_mask = np.where(mask_array > 0, 1, 0)

    # Convert back to an image
    binary_mask_img = Image.fromarray(binary_mask.astype(np.uint8))

    # Optionally, save or display the resized images
    image1_resized.save(des_full)
    binary_mask_img.save(des_mask)

for idx, f in enumerate(orig_masklist):
    des_full = task_folder + '/imagesTs/' + str(test_id[idx]) + '_0000.png'
    des_mask = task_folder + '/labelsTs/' + str(test_id[idx]) + '.png'
    resize_image(orig_filelist[idx], f, des_full, des_mask)