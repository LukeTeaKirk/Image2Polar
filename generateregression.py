import os
import cv2
import csv
import tensorflow as tf
import numpy as np
from tensorflow import keras
print(1)
fileList = os.listdir('d://intruder/training/combined/')

def cv2reader(files):
    y = []
    for x in files:
        img = cv2.imread('d://intruder/training/combined/' + x)
        resized = cv2.resize(img, (840,360), interpolation=cv2.INTER_AREA)
        y.append(resized)
    return y

def generatefilepaths():
    c = 0
    filezz = []
    while c < 10000:
        if c == 6000:
            c = 7000
        if os.path.isfile("d://intruder/training/combined/" + "image_" + str(c) + ".jpg"):
            filezz.append("image_" + str(c) + ".jpg")
            print(c)
        c = c + 1
    return filezz

def generatefilezpaths():
    c = 6000
    filezz = []
    while c < 7000:
        if os.path.isfile("d://intruder/training/combined/" + "image_" + str(c) + ".jpg"):
            filezz.append("image_" + str(c) + ".jpg")
            print(c)
        c = c + 1
    return filezz


def csvreader(count):
    with open('output.csv') as csv_file:
        csv_readerr = csv.reader(csv_file, delimiter=',')
        why = []
        for row in csv_readerr:
            roo = list(map(float, row))
            why.append(roo[1:2])
            newcount = count + 1
    return why[count:newcount], newcount


def loader(filez,batch_size):
    L = len(filez)
    #this line is just to make the generator infinite, keras needs that
    while True:
        count = 0
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = cv2reader(filez[batch_start:limit])
            Y, count = csvreader(count)
            X = np.asarray(X)
            yield X, np.asarray(Y)
            batch_start += batch_size
            batch_end += batch_size


model = tf.keras.Sequential([
    keras.layers.Conv2D(2,[3,3],1,input_shape=[360,840,3]),
    keras.layers.MaxPool2D((4,4)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=16),
    keras.layers.Dense(units=64),
    keras.layers.Dense(units=2),

])
model.compile(optimizer='adam', loss='mae', metrics=[tf.keras.metrics.Accuracy()])
fileList = generatefilepaths()
filezlist = generatefilezpaths()
model.fit(loader(fileList,1), validation_data=loader(fileList,1), steps_per_epoch=10116, epochs =2)
x = cv2.imread("D://intruder/test/image_259.jpg")
resized = cv2.resize(x, (840,360), interpolation=cv2.INTER_AREA)
resized = np.asarray(resized)
resized = resized.reshape((1, 360, 840, 3))
print(resized.shape)

print(model.predict(resized))
model.save("modelv1")
#loader()
print(2)
