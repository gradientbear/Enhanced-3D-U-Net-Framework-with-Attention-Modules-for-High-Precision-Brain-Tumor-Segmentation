from utils import plot_history, load_trained_model, evaluate_batch_IoU, visualize_predictions, imageLoader
from preprocess import OUTPUT_DIRS
import os
import json

model_history_path = os.path.join(OUTPUT_DIRS["output"], "history.json")
with open(model_history_path, 'r') as f:
    model_history = json.load(f)

# Plot training history
plot_history(model_history)
model_path = os.path.join(OUTPUT_DIRS["output"], "brats_3d.hdf5")
my_model = load_trained_model(model_path)
batch_size = 8

val_img_dir = OUTPUT_DIRS["val_images"]
val_mask_dir = OUTPUT_DIRS["val_masks"]
val_img_list = sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

test_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)
mean_iou = evaluate_batch_IoU(my_model, test_img_datagen)
print(f"Mean IoU for the batch: {mean_iou}")


# Visualize predictions for a specific test image
img_num = 97
visualize_predictions(my_model, img_num, val_img_dir, val_mask_dir)
