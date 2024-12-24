# import necessary libraries
import os
import glob
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from IPython.display import clear_output

# Path to BraTS2020 dataset
# Paths to dataset
DATASET_PATH = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
OUTPUT_DIRS = {
    "train_images": "input_data/train/images",
    "train_masks": "input_data/train/masks",
    "val_images": "input_data/val/images",
    "val_masks": "input_data/val/masks",
    "output": "output"
}
# Create output directories
for key in OUTPUT_DIRS:
    os.makedirs(OUTPUT_DIRS[key], exist_ok=True)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Load and exclude corrupted data
EXCLUDED_INDICES = [354]
def load_and_exclude_corrupted_data():
    """
    Load dataset paths and exclude corrupted data.
    """
    t1_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*t1.nii')))
    t2_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*t2.nii')))
    t1ce_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*t1ce.nii')))
    flair_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*flair.nii')))
    mask_list = sorted(glob.glob(os.path.join(DATASET_PATH, '*/*seg.nii')))
    
    # Exclude corrupted data
    for idx in EXCLUDED_INDICES:
        for lst in [t1_list, t2_list, t1ce_list, flair_list, mask_list]:
            lst.pop(idx)

    return t1_list, t2_list, t1ce_list, flair_list, mask_list

# Process images and masks
def process_image_and_mask(image_index, t1_list, t2_list, t1ce_list, flair_list, mask_list):
    """
    Process a single image and its corresponding mask.

    Args:
        image_index (int): Index of the image to process.
    """
    print(f"Processing image and mask number: {image_index}")

    # Load and normalize T2 image
    t2_image = nib.load(t2_list[image_index]).get_fdata()
    t2_image = scaler.fit_transform(t2_image.reshape(-1, t2_image.shape[-1])).reshape(t2_image.shape)

    # Load and normalize T1ce image
    t1ce_image = nib.load(t1ce_list[image_index]).get_fdata()
    t1ce_image = scaler.fit_transform(t1ce_image.reshape(-1, t1ce_image.shape[-1])).reshape(t1ce_image.shape)

    # Load and normalize Flair image
    flair_image = nib.load(flair_list[image_index]).get_fdata()
    flair_image = scaler.fit_transform(flair_image.reshape(-1, flair_image.shape[-1])).reshape(flair_image.shape)

    # Load and preprocess segmentation mask
    mask = nib.load(mask_list[image_index]).get_fdata().astype(np.uint8)
    mask[mask == 4] = 3  # Reassign label 4 to label 3

    # Normalize and crop images
    images = np.stack([flair_image, t1ce_image, t2_image], axis=-1)
    images = scaler.fit_transform(images.reshape(-1, images.shape[-1])).reshape(images.shape)
    dim = 128
    xy_start, xy_end = 184 - dim, 184 
    z_start, z_end = 141 - dim, 141
    cropped_images = images[xy_start:xy_end, xy_start:xy_end, z_start:z_end]
    cropped_mask = mask[xy_start:xy_end, xy_start:xy_end, z_start:z_end]
    cropped_mask = to_categorical(cropped_mask, num_classes=4)

    # Check if mask contains sufficient non-background labels
    val, counts = np.unique(cropped_mask, return_counts=True)
    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% of non-zero labels
        # Convert mask to one-hot encoding
        cropped_mask = to_categorical(cropped_mask, num_classes=4)

        # Save processed data
        if image_index < 150:
            print("Saving to train dataset")
            np.save(os.path.join(OUTPUT_DIRS["train_images"], f"image_{image_index}.npy"), cropped_images)
            np.save(os.path.join(OUTPUT_DIRS["train_masks"], f"mask_{image_index}.npy"), cropped_mask)
        else:
            print("Saving to validation dataset")
            np.save(os.path.join(OUTPUT_DIRS["val_images"], f"image_{image_index}.npy"), cropped_images)
            np.save(os.path.join(OUTPUT_DIRS["val_masks"], f"mask_{image_index}.npy"), cropped_mask)
    else:
        print("Image has insufficient useful data (skipped).")

# Process all images and masks
def process_data():
    """
    Process all images and masks in the dataset.
    """
    # Load and exclude corrupted data
    t1_list, t2_list, t1ce_list, flair_list, mask_list = load_and_exclude_corrupted_data()

    # Process and save all images and masks
    for img_idx in range(len(t1_list)):
        process_image_and_mask(img_idx, t1_list, t2_list, t1ce_list, flair_list, mask_list)

    # Clear the output for cleanliness
    clear_output()
    print("Data preprocessing completed.")

# Run the data preprocessing
if __name__ == "__main__":
    process_data()