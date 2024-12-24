from models import unet_model
from utils import dice_coef_loss, IoU, dice_coef, sensitivity, specificity, imageLoader
from keras.optimizers import Adam
import os
import json
import keras
from preprocess import OUTPUT_DIRS


IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes = 128, 128, 128, 3, 4
model = unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes)

# Optimizer
EPOCHS = 50
learning_rate = 1e-4
decay_rate = learning_rate / EPOCHS
opt = Adam(learning_rate=learning_rate, decay=decay_rate)

# Compile the model
model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["binary_accuracy", IoU, dice_coef, sensitivity, specificity])

# Callbacks
callbacks = [keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIRS["output"], "brats_3d.hdf5"),
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss'
                )
            ]

print(f"Model Input Shape: {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")

# Directory paths
train_img_dir = OUTPUT_DIRS["train_images"]
train_mask_dir = OUTPUT_DIRS["train_masks"]
val_img_dir = OUTPUT_DIRS["val_images"]
val_mask_dir = OUTPUT_DIRS["val_masks"]
output_dir = OUTPUT_DIRS["output"]

# Generate sorted lists of filenames for images and masks
train_img_list = sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))
val_img_list = sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

# Batch size for training and validation
batch_size = 2

# Create data generators
train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

# Steps per epoch
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

print(f"Validation Steps Per Epoch: {val_steps_per_epoch}")
print(f"Training Steps Per Epoch: {steps_per_epoch}")

# Train the model
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=50,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks
                    )

# Save the trained model
model.save(os.path.join(output_dir, "brats_3d.hdf5"))

# save model history
with open(os.path.join(output_dir, "history.json"), 'w') as f:
    json.dump(history.history, f)


