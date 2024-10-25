import numpy as np
from skimage.io import imread
import os

def dice_score(y_true, y_pred):
    """
    Calculate the Dice score between two binary masks.
    Dice = 2 * (intersection) / (sum of pixels in both images)
    """
    intersection = np.sum(y_true * y_pred)
    return 2 * intersection / (np.sum(y_true) + np.sum(y_pred))

def calculate_dice_for_pngs(ground_truth_folder, prediction_folder):
    """
    Calculates the Dice score for pairs of ground truth and prediction PNG images.
    
    :param ground_truth_folder: Folder containing the ground truth PNG images.
    :param prediction_folder: Folder containing the predicted PNG images.
    """
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith('.png')])
    prediction_files = sorted([f for f in os.listdir(prediction_folder) if f.endswith('.png')])
    
    dice_scores = []
    
    for gt_file, pred_file in zip(ground_truth_files, prediction_files):
        # Load the ground truth and predicted images
        y_true = imread(os.path.join(ground_truth_folder, gt_file), as_gray=True)
        y_pred = imread(os.path.join(prediction_folder, pred_file), as_gray=True)
        
        # Ensure the images are binary (0 and 1 values)
        y_true = (y_true > 0.5).astype(np.uint8)
        y_pred = (y_pred > 0.5).astype(np.uint8)
        
        # Calculate the Dice score for each pair
        score = dice_score(y_true, y_pred)
        dice_scores.append(score)
        print(f"Dice score for {gt_file} and {pred_file}: {score:.4f}")
    
    # Return the average Dice score over all images
    avg_dice = np.mean(dice_scores)
    median_dice = np.median(dice_scores)
    print(f"Average Dice score: {avg_dice:.4f}")
    print(f"Median Dice: {median_dice:.4f}")
    return avg_dice, median_dice

# Example usage
ground_truth_folder = '/research/m323170/Projects/mammography/dataset/nnUNet_raw/Dataset033_Mammography033_CBIS/labelsTs/'
prediction_folder = '/research/m323170/Projects/mammography/dataset/postprocessing/Dataset033_Mammography033_CBIS/'
calculate_dice_for_pngs(ground_truth_folder, prediction_folder)

