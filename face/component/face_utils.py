
import cv2
import os
import shutil

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def is_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def get_cv2_classifier(config):
    return cv2.CascadeClassifier(config.get_face_classifier_file())

def delete_face_folder(config, name):
    del_path = os.path.join(config.get_work_folder(), name)
    print("del_path", del_path)
    shutil.rmtree(del_path, ignore_errors=True)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0) 
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        return image

    def get_frame_jpg_bytes(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()