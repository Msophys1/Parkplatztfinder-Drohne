import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load data from a CSV file
data = pd.read_csv('_annotations.csv')

# One-hot encode the class labels
class_labels = pd.get_dummies(data['class'])

# Create lists to store image and bounding box data
data_list = []

# Load images and bounding box data
for index, row in data.iterrows():
    # Load the image from the corresponding folder
    img = Image.open(os.path.join('data', row['filename']))
    # Resize the image to 224 x 224
    image = img.resize((224, 224))
    # Get the dimensions of the image
    img_width, img_height = image.size
    # Normalize the pixel values of the image
    nm_image = np.array(image) / 255.
    # Subtract the mean and divide by the standard deviation of the pixels
    dev_image = (nm_image - np.mean(nm_image)) / np.std(nm_image)
    # Extract bounding box coordinates and normalize them
    xmin = row['xmin'] / img_width
    ymin = row['ymin'] / img_height
    xmax = row['xmax'] / img_width
    ymax = row['ymax'] / img_height
    # Add bounding box coordinates and class label to the list as a tuple
    data_list.append((dev_image, (xmin, ymin, xmax, ymax)))

# Convert list of tuples to numpy array
data_array = np.array(data_list, dtype=object)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_array[:, 0], class_labels, test_size=0.2, random_state=42,
                                                    shuffle=True)

# Convert class labels to numpy arrays
y_train = np.array(y_train, dtype=float)
y_test = np.array(y_test, dtype=float)

X_train_tensor = np.stack(X_train[:], axis=0)
y_train_tensor = y_train
X_test_tensor = np.stack(X_test[:], axis=0)
y_test_tensor = y_test

X_train_tensor = tf.convert_to_tensor(X_train_tensor, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train_tensor, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_tensor, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test_tensor, dtype=tf.float32)


def get_model():
    # Load the pre-trained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Freeze all layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    # Compile the model with binary cross-entropy loss and Adam optimizer
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


model = get_model()

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train_tensor, y_train_tensor, epochs=50, validation_data=(X_test_tensor, y_test_tensor),
          callbacks=[early_stopping])

# Save the model in SavedModel format
model.save('my_model.h5', include_optimizer=True)