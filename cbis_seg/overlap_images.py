import re
import os
from shutil import copyfile
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import numpy as np

dir_path='/research/m323170/Projects/mammography'

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
def overlap_img(base_mask, append_mask, file_path):
    append_mask = f"{dir_path}/{append_mask}"
    print(base_mask)
    base_image = Image.open(base_mask)
    mask_image = Image.open(append_mask)
    # Resize the mask to match the size of the base image (if necessary)
    mask_image_resized = mask_image.resize(base_image.size)
    # Convert mask to binary (ensure mask values are only 0 and 1)
    binary_mask = np.array(mask_image_resized) > 0  # Convert mask to binary: 0 or 1
    binary_mask = binary_mask.astype(np.uint8)  # Ensure binary mask has integer values
    base_mask = np.array(base_image) > 0  # Convert mask to binary: 0 or 1
    base_mask = base_mask.astype(np.uint8)  # Ensure binary mask has integer values
    merged_mask = np.logical_or(base_mask, binary_mask)

    # Convert boolean mask to integer (0 and 1)
    merged_mask = merged_mask.astype(int)

    # Convert back to an image
    overlayed_image = Image.fromarray(merged_mask.astype(np.uint8))

    # Show or save the result
    #overlayed_image.show()
    overlayed_image.save(file_path)
    # Plot the image using matplotlib
    #plt.imshow(overlayed_image, cmap='gray')  # Use 'gray' colormap for a binary image
    #plt.title('Binary Mask')
    #plt.axis('off')  # Hide axis for better visualization
    #plt.show()
def find_duplicates(input_list):
    seen = set()       # To store unique elements
    duplicates = set()  # To store duplicates

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)
for idx in range(len(train)):
    SeriesUID_mask = train_list[idx]['ROI mask file path'].str.extract(re.escape(train[idx])+r'.*\/.*\/(.*)\/')
    SeriesUID = train_list[idx]['image file path'].str.extract(re.escape(train[idx])+r'.*\/.*\/(.*)\/')
    path_list = []
    assert len(SeriesUID_mask) == len(SeriesUID)
    for ipath in range(len(SeriesUID)):
        if SeriesUID_mask[0][ipath] in dicom_mask.SeriesInstanceUID.values:
            path_list.append(dicom_mask[dicom_mask['SeriesInstanceUID'] == SeriesUID_mask[0][ipath]].image_path.item())
        else:
            print('[train]mask not exist')
            print(f"full:{SeriesUID[0][ipath]}")
            print(f"mask:{SeriesUID_mask[0][ipath]}")
            train_list[idx].drop([ipath], inplace=True)
    train_list[idx]['roi_mask_path']=path_list
    dataset=train_list[idx]
for idx in range(len(train)):
    SeriesUID = train_list[idx]['image file path'].str.extract(re.escape(train[idx])+r'.*\/.*\/(.*)\/')
    duplicates = find_duplicates(train_list[idx]['image file path'])
    #print("Duplicates:", duplicates)
    for full_img in duplicates:
        result = train_list[idx].loc[train_list[idx]['image file path'] == full_img, 'roi_mask_path']
        s=re.compile('.*\/.*\/(.*)\/')
        id = s.findall(full_img)[0]
        file_path = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset077_Mammography077_CBIS/labelsTr/'+id+'.png'
        #print(f"===full_img:{id}")
        #print(result)
        for iimg, img in enumerate(result):
            if iimg == 0:
                base_img = f"{dir_path}/{img}"
                continue
            overlap_img(base_img, img, file_path)
            base_img=file_path