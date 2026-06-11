import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = "C:\\Users\\Aadarsh\\Desktop\\training\\Harmful Object Detection\\train\\data"
test_dir = "C:\\Users\\Aadarsh\\Desktop\\training\\Harmful Object Detection\\Test\\Data"

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')

test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')

# ==================== MODEL ====================
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # freeze base

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==================== TRAIN ====================
model.fit(train_gen, validation_data=test_gen, epochs=10)

# ==================== SAVE ====================
model.save("model.h5")

# Save class labels in same order
labels = list(train_gen.class_indices.keys())
with open("knife.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

print("✅ Model and labels saved!")
