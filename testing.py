import xml.etree.ElementTree as ET
import cv2
import numpy
import pandas as pd
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, GlobalMaxPooling2D

# Define data directories
data_dir = "data"
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


train_images = tf.data.Dataset.list_files('data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x / 255)
test_images = tf.data.Dataset.list_files('data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = test_images.map(lambda x: x / 255)
val_images = tf.data.Dataset.list_files('data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = val_images.map(lambda x: x / 255)

# Build label loading function

label_map = {'cat': '0', 'drone': '1', 'drones': '2', 'cucumber': '3', 'nothing': "4", 'girafe': '5'}
def load_labels(label_path):
    with open(label_path.numpy().decode('utf-8'), 'r') as t:
        label_tree = ET.parse(t)
        label_root = label_tree.getroot()
        label_class = label_root.find('object').find('name').text
        label_class = label_map[str(label_class)]
        label_bbox = [float(label_root.find('object').find('bndbox').find('xmin').text),
                      float(label_root.find('object').find('bndbox').find('ymin').text),
                      float(label_root.find('object').find('bndbox').find('xmax').text),
                      float(label_root.find('object').find('bndbox').find('ymax').text)]
    return label_class, label_bbox


train_labels = tf.data.Dataset.list_files('data/train/label/*.xml', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.string, tf.float32]))
test_labels = tf.data.Dataset.list_files('data/test/label/*.xml', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.string, tf.float32]))
val_labels = tf.data.Dataset.list_files('data/val/label/*.xml', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.string, tf.float32]))

# train_labels.as_numpy_iterator().next()

# Building of the end dataset

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

# var = train.as_numpy_iterator().next()[1]
# print(var)

# get a data sample

"""data_samples = train.as_numpy_iterator()
res = data_samples.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    ax[idx].imshow(sample_image)"""

# download VGG16

vgg = VGG16(include_top=False)
vgg.summary()


# Build instance of Network
def build_model():
    input_layer = Input(shape=(120, 120, 3))
    vgg = VGG16(include_top=False)(input_layer)  # don't include the firsts top layer
    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    return Model(inputs=input_layer, outputs=[class2, regress2])


# Test out Neural Network
drone_tracker = build_model()
drone_tracker.summary()
X, y = train.as_numpy_iterator().next()
X.shape
classes, coords = drone_tracker.predict(X)
classes, coords

# define loss and optimizer
batches_per_epoch = len(train)
lr_decay = (1. / 0.75 - 1) / batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


# Create Localization Loss and Classification Loss
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))
    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]
    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size


classloss = tf.keras.losses.BinaryCrossentropy
regressloss = localization_loss

# Test out Loss Metrics

localization_loss(y[1], coords)
classloss(y[0], classes)
regressloss(y[1], coords)


# Train Create Custom Model Class
class Drone_Tracker(Model):
    def __init__(self, track, **kwargs):
        super().__init__(**kwargs)
        self.model = track

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_classloss = self.closs(tf.cast(y[0], tf.float32), classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            total_loss = (batch_localizationloss + 0.5 * batch_classloss)
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch
        classes, coords = self.model(X, training=False)
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


model = Drone_Tracker(drone_tracker)
model.compile(opt, classloss, regressloss)

# Train

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback])

# plot performances

var = hist.history
print(var)
fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

# Make predictions

test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = drone_tracker.predict(test_sample[0])
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

    ax[idx].imshow(sample_image)

