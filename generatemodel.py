import tensorflow as tf
import os
import scipy
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

far_dir = os.path.join('training/far/')
near_dir = os.path.join('training/near/')
middle_dir = os.path.join('training/middle/')

print('total training far images:', len(os.listdir(far_dir)))
print('total training near images:', len(os.listdir(near_dir)))
print('total training middle images:', len(os.listdir(middle_dir)))

far_files = os.listdir(far_dir)
print(far_files[:10])

near_files = os.listdir(near_dir)
print(near_files[:10])

middle_files = os.listdir(middle_dir)
print(middle_files[:10])
TRAINING_DIR = "training/"
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    )
print (1)
VALIDATION_DIR = "validation/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(1920,1080),
    color_mode='grayscale',
	class_mode='categorical',
    batch_size=1
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(1920,1080),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=5
)
print (2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (8,8), activation='relu', input_shape=(1920, 1080, 1)),
    tf.keras.layers.MaxPooling2D(6, 6),
    #tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

print (3)

model.summary()
print (4)

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print (5)
callbacks = myCallback()

history = model.fit(train_generator, epochs=4, steps_per_epoch=200, validation_data = validation_generator, verbose = 1, validation_steps=25, callbacks=[callbacks])
print (6)

model.save("rps.h5")
