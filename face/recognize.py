from flask import Blueprint, Response, flash, request, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename
from .sysconfig import get_sysconfig
from .component.face_utils import VideoCamera
from os.path import join

import os
import cv2
import time
import traceback
import uuid

bp = Blueprint('recognize', __name__, url_prefix='/recognize')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

@bp.route('/')
def index():
    return "detection"

@bp.route('/upload/', methods=['post'])
def upload():
    config = get_sysconfig()
    
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        #filename = secure_filename(file.filename)
        
        # rename the file name by UUID
        filename, file_extension = os.path.splitext(file.filename)
        filename = str(uuid.uuid4())+ file_extension
        uploaded_image_file = os.path.join(config.get_temp_folder(), filename)
        print("uploaded_image_file:",uploaded_image_file)
        
        file.save(uploaded_image_file)

        recognize_who_i_am(config, uploaded_image_file)

        return redirect(url_for('recognize.uploaded_file',
                                filename=filename))
    return "upload image"

@bp.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    config = get_sysconfig()
    return send_from_directory(config.get_temp_folder(),filename)

@bp.route('/camera/<name>')
def camera(name):
    config = get_sysconfig()
    return Response(gen(config, VideoCamera(),name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')   

def gen(config, camera, name):
    print(config.get_face_classifier_file())
    face_classifier = cv2.CascadeClassifier(config.get_face_classifier_file())
    tainedFile = join(config.get_work_folder(),name,"trained_result.yml")

    display_string = ""
    while True:
        image, face = face_detector(face_classifier, camera.get_frame())

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(tainedFile)
            results = model.predict(face)
            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'

        except:
            display_string = "No face found"
            pass

        cv2.putText(image, display_string, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        time.sleep(0.05)

# Load functions
def face_detector(face_classifier, img):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_who_i_am(config , pic_path, is_mark_on_pic=True):
    face_classifier = cv2.CascadeClassifier(config.get_face_classifier_file())    
    img=cv2.imread(pic_path)
    
    image, face = face_detector(face_classifier, img)
    print(len(face))
    if len(face) == 0:
        return

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    names = get_names(config)

    recog_result = {
        'confidence':0,
        'name':'unknow'
        }

    for name in names:
        confidence = 0    
        try:
            trained_file = join(config.get_work_folder(),name,'trained_result.yml')
            if os.path.exists(trained_file) == False:
                continue
            print(trained_file)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(trained_file)

            # Pass face to prediction model
            # "results" comprises of a tuple containing the label and the confidence value
            results = model.predict(face)

            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
            else :
                print('confidence is low')

            print('name:%s, confidence:%d' % (name,confidence))

            if recog_result['confidence'] < confidence:
                recog_result['confidence'] = confidence
                recog_result['name'] = name
        except:
            traceback.print_exc()
            continue
        
    display_string = str(recog_result['confidence']) + '% Confidence is ' + recog_result['name']
    if is_mark_on_pic:
        cv2.putText(image, display_string, (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
        #cv2.putText(image, recog_result['name'] , (30,60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imwrite(pic_path,img)

    return display_string

def get_names(configs):
    dirList = os.listdir(configs.get_work_folder())
    faces = [dir for dir in dirList if os.path.isdir(os.path.join(configs.get_work_folder(), dir)) == True]

    return faces
