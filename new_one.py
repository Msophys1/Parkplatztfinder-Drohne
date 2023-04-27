import os
import re
import pandas as pd
import tensorflow as tf
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from tensorflow import data, io, keras

# Define hyper-parameters
image_size = (224, 224)
batch_size = 32
epochs = 10
train_images_tfrecord_dir = 'Data1/train'
validation_images_tfrecord_dir = 'Data1/valid'
test_images_tfrecord_dir = 'Data1/test'
num_classes = 10


# Load the labels from the pbtxt file
def get_labels_from_pbtxt(pbtxt_file):
    labels = {}
    with open(pbtxt_file, 'r') as f:
        pbtxt = f.read()
        items = re.findall('item\s*{([^}]*)}', pbtxt)
        for item in items:
            name = re.search('name:\s*[\'"]([^\'"]*)[\'"]', item).group(1)
            display_name = re.search('display_name:\s*[\'"]([^\'"]*)[\'"]', item).group(1)
            id = int(re.search('id:\s*([0-9]+)', item).group(1))
            label = pd.get_dummies(name).values.tolist()[0]
            labels[id] = {'name': name, 'display_name': display_name, 'label': label}
    return labels


# Define the model
base_model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile the model
loss_fn = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.Adam()
metrics = [keras.metrics.CategoricalAccuracy()]
model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

# Create train dataset from TFRecord files
train_image_files = io.gfile.glob(os.path.join(train_images_tfrecord_dir, '*.tfrecord'))
train_dataset = data.TFRecordDataset(train_image_files)
train_dataset = train_dataset.map(lambda x: tf.io.parse_single_example(x, features={
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}))
train_labels = get_labels_from_pbtxt('Data1/train/drones_label_map.pbtxt')
train_dataset = train_dataset.map(lambda x: (tf.image.decode_jpeg(x['image'], channels=3), train_labels))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Create validation dataset from TFRecord files
validation_image_files = io.gfile.glob(os.path.join(validation_images_tfrecord_dir, '*.tfrecord'))
validation_dataset = data.TFRecordDataset(validation_image_files)
validation_dataset = validation_dataset.map(lambda x: tf.io.parse_single_example(x, features={
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}))
validation_labels = get_labels_from_pbtxt('Data1/valid/drones_label_map.pbtxt')
validation_dataset = validation_dataset.map(lambda x: (tf.image.decode_jpeg(x['image'], channels=3), validation_labels))
validation_dataset = validation_dataset.batch(batch_size)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Create test dataset from TFRecord files
test_image_files = io.gfile.glob(os.path.join(test_images_tfrecord_dir, '*.tfrecord'))
test_dataset = data.TFRecordDataset(test_image_files)
test_dataset = test_dataset.map(lambda x: tf.io.parse_single_example(x, features={
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}))
test_labels = get_labels_from_pbtxt('Data1/test/drones_label_map.pbtxt')
test_dataset = test_dataset.map(lambda x: (tf.image.decode_jpeg(x['image'], channels=3), test_labels))
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Train the model with mapped datasets
history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, batch_size=batch_size)
