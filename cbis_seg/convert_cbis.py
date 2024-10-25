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
train = ['Mass-Training', 'Calc-Training']
test = ['Mass-Test', 'Calc-Test']
train_list = [mass_train, calc_train]
test_list = [mass_test, calc_test]
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
assert len(orig_masklist) == len(orig_filelist)

def overlay_image():
    # Load the base grayscale image and the binary mask
    base_image = Image.open(f'{dir_path}/CBIS-DDSM/jpeg/1.3.6.1.4.1.9590.100.1.2.357214585012696579228748059281324511668/2-072.jpg')
    mask_image = Image.open(f'{dir_path}/CBIS-DDSM/jpeg/1.3.6.1.4.1.9590.100.1.2.331828654011758947930839827374231003829/2-236.jpg')

    # Resize the mask to match the size of the base image (if necessary)
    mask_image_resized = mask_image.resize(base_image.size)

    # Convert mask to binary (ensure mask values are only 0 and 1)
    binary_mask = np.array(mask_image_resized) > 0  # Convert mask to binary: 0 or 1
    binary_mask = binary_mask.astype(np.uint8)  # Ensure binary mask has integer values

    # Overlay the binary mask on the base image (mask=1 will keep mask, mask=0 will keep base image)
    overlayed_image_array = np.array(base_image) * (1 - binary_mask) + binary_mask * 255
    overlayed_image_array = np.where(overlayed_image_array > 0, 1, 0)

    # Convert back to an image
    overlayed_image = Image.fromarray(overlayed_image_array.astype(np.uint8))

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
    des_full = task_folder + '/imagesTr/' + str(train_id[idx]) + '_0000.png'
    des_mask = task_folder + '/labelsTr/' + str(train_id[idx]) + '.png'
    resize_image(orig_filelist[idx], f, des_full, des_mask)
test_id = []
orig_filelist = []
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
        test_id.append(SeriesUID[0][ipath])
    dataset=test_list[idx]
assert len(orig_filelist) == len(test_id)
print(f"Test: {len(SeriesUID)}, {len(orig_filelist)}")
for idx, f in enumerate(orig_filelist):
    try: 
        copyfile(f, task_folder + '/imagesTs/' + str(test_id[idx]) + '_0000.png')
    except Exception as e:
        print(e)
n_train=len(orig_masklist)
n_test=len(orig_filelist)

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
json_dict['file_ending'] = ".png"
json_dict['numTest'] = n_test

json_dict['training'] = [
        {'image': "./imagesTr/" +  train_id[idx] + '_0000.png', "label": "./labelsTr/" +  train_id[idx] + '.png'} for idx in range(len(train_id))]
json_dict['test'] = ["./imagesTs/" + test_id[ipath]+ '_0000.png' for ipath in range(len(test_id))]

save_json(json_dict, task_folder + "dataset.json")