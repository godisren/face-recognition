import os
import click
import cv2

from flask import Flask, current_app as app, g
from flask.cli import with_appcontext
from . import sysconfig
from .component.face_utils import is_image_file
from .component import face_utils
from . import sample
from . import train


def init_app(app):
    #app.teardown_appcontext(close_db)
    app.cli.add_command(recognize_all_command)
    app.cli.add_command(sample_command)
    app.cli.add_command(train_command)
    app.cli.add_command(delete_command)
    

@click.command('recognize-all')
@click.argument('file_path')
@with_appcontext
def recognize_all_command(file_path):
    click.echo('recognize picture.')

    if os.path.isfile(file_path) == False :
        print("File not found.", file_path)
        return

    from . import recognize
    result = recognize.recognize_who_i_am(sysconfig.get_sysconfig(),file_path,False)
    print("recognized result:", result)

@click.command('sample')
@click.argument('name')
@click.argument('image_folder')
@with_appcontext
def sample_command(name, image_folder):
    click.echo('sampling picture from '+ image_folder+ ' for ' + name)

    if not os.path.isdir(image_folder) :
        click.echo('folder not found')
        return

    #faces = [dir for dir in dirList if os.path.isdir(os.path.join(configs.get_work_folder(), dir)) == True]
    images = [ img_path for img_path in os.listdir(image_folder) if is_image_file(os.path.join(image_folder,img_path)) == True ]

    config = sysconfig.get_sysconfig()
    face_classifier = face_utils.get_cv2_classifier(config)

    for idx, img_path in enumerate(images):
        img =cv2.imread(os.path.join(image_folder,img_path))
        extract_face, faces_range = sample.face_extractor_by_cropping(face_classifier, img)
        if extract_face is not None:
            sample.save_image_by_cropping(config, name, idx+1, extract_face)
        else:
            # Face not found
            pass

@click.command('train')
@click.argument('name')
@with_appcontext
def train_command(name):
    click.echo('training picture for ' + name)
    config = sysconfig.get_sysconfig()
    train.training(config, name)

    click.echo('training success.')

@click.command('delete')
@click.argument('name')
@with_appcontext
def delete_command(name):
    click.echo('deleting picture for ' + name)

    face_utils.delete_face_folder(sysconfig.get_sysconfig(), name)

    click.echo('deleting success.')

