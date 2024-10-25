import re
import os
from shutil import copyfile
import pandas as pd

import os
import nibabel as nib
import numpy as np
from PIL import Image

def convert_pngs_to_nifti(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Load the PNG image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            image_np = np.array(image)

            # Convert to NIfTI
            nifti_image = nib.Nifti1Image(image_np, affine=np.eye(4))

            # Save as NIfTI file in the output folder
            nifti_filename = os.path.splitext(filename)[0] + '.nii.gz'
            output_path = os.path.join(output_folder, nifti_filename)
            nib.save(nifti_image, output_path)

            print(f"Converted {filename} to {nifti_filename}")

# Set your input and output directories
input_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset077_Mammography077_CBIS/labelsTr/'  # Replace with the path to your PNG folder
output_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset088_Mammography088_CBIS_Nifti/labelsTr/'  # Replace with the desired output folder

# Run the conversion
convert_pngs_to_nifti(input_folder, output_folder)
