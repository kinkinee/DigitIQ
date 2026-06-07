import tensorflow as tf
import numpy as np
import os
import cv2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

custom_images = []
custom_labels = []

for digit in range(10):
    folder = f"my_digits/{digit}"
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Invert (MNIST style)
        img = 255 - img
        # Threshold
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(img)
        if coords is None:
            continue
        # Crop digit
        x, y, w, h = cv2.boundingRect(coords)
        
        img = img[y:y+h, x:x+w]
        h, w = img.shape

        size = max(h, w)

        square = np.zeros(
        (size, size),
        dtype=np.uint8)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2

        square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
        ] = img

        square = cv2.copyMakeBorder(
        square, 4, 4, 4, 4,
        cv2.BORDER_CONSTANT,
        value=0)

        img = cv2.resize(
        square,
        (28,28))

        # Normalize
        img = img / 255.0

        custom_images.append(img)
        custom_labels.append(digit)

custom_images = np.array(custom_images)
custom_labels = np.array(custom_labels)

custom_images = custom_images.reshape(-1, 28, 28, 1)
custom_images = custom_images.reshape(-1, 28, 28, 1)

combined_x = np.concatenate([x_train, custom_images], axis=0)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

combined_y = np.concatenate([y_train, custom_labels], axis=0)

print("Custom images:", custom_images.shape)
print("Custom labels:", custom_labels.shape)

print(combined_x.shape)
print(combined_y.shape)

model = tf.keras.models.load_model("digit_cnn.keras")
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(combined_x)

model.fit(
    datagen.flow(
        combined_x,
        combined_y,
        batch_size=32
    ),
    epochs=5
)

model.save("digit_cnn_combined.keras")

predictions = model.predict(custom_images)

correct = 0

for i in range(len(custom_labels)):
    pred = np.argmax(predictions[i])

    if pred == custom_labels[i]:
        correct += 1
        f"Actual: {custom_labels[i]}  Predicted: {pred}"
    

print(
    "Custom Accuracy:",
    correct / len(custom_labels)
)