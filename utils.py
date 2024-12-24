# import necessary libraries
import os
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU

def dice_coef(y_true, y_pred, smooth=1):
    """
    Compute the Dice coefficient for model evaluation.

    Args:
        y_true (tf.Tensor): Target tensor.
        y_pred (tf.Tensor): Predicted tensor.
        smooth (float): Smoothing factor.

    Returns:
        tf.Tensor: Dice coefficient score.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def sensitivity(y_true, y_pred):
    """
    Compute sensitivity for binary classification.

    Args:
        y_true (tf.Tensor): Target tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: Sensitivity score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    """
    Compute specificity for binary classification.

    Args:
        y_true (tf.Tensor): Target tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: Specificity score
    """
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def IoU(y_true, y_pred):
    """
    Compute Intersection over Union (IoU) for predicted and target masks.
    
    Args:
        y_true (tf.Tensor): Target masks.
        y_pred (tf.Tensor): Predicted masks.

    Returns:
        tf.Tensor: IoU score.
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    return K.mean(intersection / union)

def load_img(img_dir, img_list):
    """
    Load a batch of images or masks from the specified directory.

    Args:
        img_dir (str): Directory path where images/masks are stored.
        img_list (list): List of image/mask filenames to load.

    Returns:
        np.ndarray: Array of loaded images/masks.
    """
    images = []
    for image_name in img_list:
        if image_name.endswith('.npy'):  # Only process .npy files
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    return np.array(images)

# Generator function to load data in batches
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    A data generator function to load images and masks in batches for training.

    Args:
        img_dir (str): Directory containing the input images.
        img_list (list): List of filenames for input images.
        mask_dir (str): Directory containing the ground truth masks.
        mask_list (list): List of filenames for masks.
        batch_size (int): Number of samples per batch.

    Yields:
        tuple: A batch of (images, masks) as numpy arrays.
    """
    L = len(img_list)  # Total number of samples

    # Infinite loop for data generation (required by Keras)
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            # Load a batch of images and masks
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # Return a tuple of (images, masks)
            batch_start += batch_size
            batch_end += batch_size

# Function to plot training history
def plot_history(history):
    """
    Plot training and validation metrics from model history.

    Args:
        history: History object from model training.
    """
    # Plot loss, accuracy, IoU, dice_coef, sensitivity, specificity
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, metric in enumerate(['loss', 'binary_accuracy', 'IoU', 'dice_coef', 'sensitivity', 'specificity']):
        axes[i].plot(history.history[metric], label='Train')
        axes[i].plot(history.history['val_' + metric], label='Validation')
        axes[i].set_title(f'Model {metric}')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('Epochs')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('plot.png')

# Custom loss and metrics for model loading
custom_objects = {
    'dice_coef_loss': dice_coef_loss,
    'IoU': IoU,
    'dice_coef': dice_coef,
    'sensitivity': sensitivity,
    'specificity': specificity
}

# Load the trained model for evaluation
def load_trained_model(model_path, custom_objects=custom_objects):
    """
    Load the trained model with custom loss and metrics.

    Args:
        model_path (str): Path to the saved model file.
        custom_objects (dict): Dictionary of custom loss and metrics.

    Returns:
        model: Loaded Keras model.
    """
    return load_model(model_path, custom_objects=custom_objects)

# Evaluate IoU on a batch of test images
def evaluate_batch_IoU(model, test_img_datagen, num_classes=4):
    """
    Evaluate Mean IoU for a batch of test images.

    Args:
        model: Trained model to evaluate.
        test_img_datagen: Data generator for test images.
        num_classes (int): Number of classes in segmentation.

    Returns:
        float: Mean IoU value for the batch.
    """
    test_image_batch, test_mask_batch = test_img_datagen.__next__()
    test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
    test_pred_batch = model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

    iou_metric = MeanIoU(num_classes=num_classes)
    iou_metric.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
    return iou_metric.result().numpy()

# Visualize predictions on a single test image
def visualize_predictions(model, img_num, img_dir, mask_dir):
    """
    Visualize predictions for a single test image.

    Args:
        model: Trained model for prediction.
        img_num (int): Index of the test image to visualize.
        img_dir (str): Directory containing test images.
        mask_dir (str): Directory containing test masks.
    """
    # Load test image and mask
    test_img = np.load(f"{img_dir}/image_{img_num}.npy")
    test_mask = np.load(f"{mask_dir}/mask_{img_num}.npy")
    test_mask_argmax = np.argmax(test_mask, axis=3)

    # Expand dimensions for model prediction
    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = model.predict(test_img_input)
    test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

    # Visualize slices
    slice_idx = 30
    for i in range(7):
        slice_idx += 10
        plt.figure(figsize=(12, 8))
        plt.subplot(131)
        plt.title('Testing Image')
        plt.imshow(test_img[:, :, slice_idx, 1], cmap='gray')
        plt.subplot(132)
        plt.title('Ground Truth Label')
        plt.imshow(test_mask_argmax[:, :, slice_idx])
        plt.subplot(133)
        plt.title('Predicted Label')
        plt.imshow(test_prediction_argmax[:, :, slice_idx])
        plt.show()

