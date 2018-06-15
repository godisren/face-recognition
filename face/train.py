from flask import Blueprint
from os import listdir
from os.path import isfile, join
from .sysconfig import get_sysconfig

import os
import cv2
import numpy as np

bp = Blueprint('train', __name__, url_prefix='/train')

@bp.route('/')
def index():
    return "training"

@bp.route('/execute/<name>')
def execute(name):
    config = get_sysconfig()
    tained_file_path = training(config, name)
    return tained_file_path

def training(config, name):
    # Get the training data we previously made
    data_path = join(config.get_work_folder(), name, "samples" + os.sep)
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    #model = cv2.createLBPHFaceRecognizer()
    #model = cv2.face.createLBPHFaceRecognizer()
    model = cv2.face.LBPHFaceRecognizer_create()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

    # Let's train our model 
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")

    # save training data
    trained_path = join(config.get_work_folder(),name,'trained_result.yml')
    print(trained_path)
    model.save(trained_path)

    return trained_path