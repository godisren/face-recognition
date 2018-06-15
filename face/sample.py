from flask import Blueprint, request, Response
from .sysconfig import get_sysconfig
from .component.face_utils import VideoCamera
from .component import face_utils

import cv2
import os

bp = Blueprint('sample', __name__, url_prefix='/sample')

@bp.route('/')
def index():
    return "sampling"

@bp.route('/upload/<name>')
def upload(name):
    return "upload images"

@bp.route('/camera/<name>')
def camera(name):
    config = get_sysconfig()
    return Response(gen(config, VideoCamera(),name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(config, camera, name):
    face_classifier = face_utils.get_cv2_classifier(config)
    #face_classifier = cv2.CascadeClassifier(config.get_face_classifier_file())

    count = 1
    while True:
        frame = camera.get_frame()
        extract_face, faces_range = face_extractor_by_cropping(face_classifier, frame)
        if extract_face is not None:
            
            save_image_by_cropping(config, name, count, extract_face)

            #file_name_path = os.path.join("D:\\mine\\learning\\python_OpenCV\\flask\\sample_pic\\test" , str(count) + '.jpg')
            #cv2.imwrite(file_name_path, frame)

            # print screen
            frame = cv2.rectangle(frame,(faces_range[0],faces_range[1]),(faces_range[2],faces_range[3]),(0,255,255),2)
            cv2.putText(frame, str(count), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            count += 1
            
            if count == 101:
                break
        else:
            # Face not found
            pass

def save_image_by_cropping(config, name, count, extract_face):
    face = cv2.resize(extract_face, (200, 200))
    # Put count on images and display live count
    cv2.putText(face, str(count), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    
    # convert to gray
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Save file in specified directory with unique name
    file_name_path = os.path.join(config.get_work_folder() , name , "samples", str(count) + '.jpg')
    os.makedirs(os.path.dirname(file_name_path), exist_ok=True)
    cv2.imwrite(file_name_path, face)

    return face

# Load functions
def face_extractor_by_cropping(face_classifier, img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None, None
    
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        faces_range = (x,y,x+w,y+h)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face, faces_range