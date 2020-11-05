import os
import pathlib
import csv
import cv2
import matplotlib
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

# the runtime initialization will not allocate all memory on the device to avoid out of GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpu,
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=4000)])

def loop():
    global count
    count = 4999
    while count < 12554:
        count = count + 1
        checkCount()
def checkCount():
    global count

    while not os.path.isfile('training/combined/image_' + str(count) + '.jpg'):
        print("skipping")
        count = count + 1
    ImageInput()

def ImageInput():
    global img
    global count
    img = cv2.imread('training/combined/image_' + str(count) + '.jpg')
    img2 = cv2.imread('training/combined/image_' + str(count) + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_resized = tf.image.resize(img, [384, 384], method='bicubic', preserve_aspect_ratio=False)
    img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    global img2np
    img2np = np.array(img2)
    reshaper_img = img2.reshape(1, img2np.shape[0], img2np.shape[1], 3)
    reshape_img = img_input.reshape(1, 3, 384, 384)
    global tensor
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)
    global image_tensor
    image_tensor = tf.convert_to_tensor(reshaper_img, dtype=tf.uint8)
    GetAngle()

def loadlibraries():
    print("load")
    global module2
    global module
    module2 = hub.load("rcnn/rcnn")
    module = hub.load("midas/", tags=['serve'])
    print("done")
def GetAngle():
    result = module2(image_tensor)
    class_ids = result["detection_classes"]
    detectionboxes = result["detection_boxes"]
    global label_id_offset
    label_id_offset = 0
    global image_np_with_detections
    image_np_with_detections = img2np.copy()
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    indexpos = -1
    for x in class_ids:
        for y in x:
            indexpos = indexpos + 1
            if y == 1:
                break
    detectionboxes = detectionboxes.numpy()
    FinalCoordinate = detectionboxes[0,indexpos]
    global FinalYCoor
    global FinalXCoor
    FinalYCoor = (FinalCoordinate[0] + FinalCoordinate[2])/2
    FinalXCoor = (FinalCoordinate[1] + FinalCoordinate[3])/2
    GetDepthMap()




def GetDepthMap():
    output = module.signatures['serving_default'](tensor)
    prediction = output['default'].numpy()
    prediction = prediction.reshape(384, 384)

# output file
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    depth_min = prediction.min()
    depth_max = prediction.max()
    img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
    global Depth
    Depth = img_out[int(FinalYCoor*img.shape[0]), int(FinalXCoor*img.shape[1])]
    Depth = Depth/256
    Depth = -1.7*Depth + 1.7
    if(Depth > 1):
        Depth = 1
    if(Depth < 0.1):
        Depth = 0.1
    print(Depth)
    getOffsetAngle()


def getOffsetAngle():
    global BaseAngle
    BaseAngle = (1-FinalXCoor)*120
    BaseAngle = int(BaseAngle)
    print(BaseAngle)
    outputData()



def outputData():
    fields = [count, Depth, BaseAngle]
    with open('output.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


loadlibraries()
loop()
