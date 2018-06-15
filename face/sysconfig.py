import os
import tempfile
import click

from flask import Flask, current_app, g
from flask.cli import with_appcontext


class SysConfig:
    def __init__(self, app):
        self.configs = {
            'WORK_FOLDER' : app.config['WORK_FOLDER'],            
            'TEMP_FOLDER' : tempfile.gettempdir(),            
            'FACE_CLASSIFIER_FILE' : os.path.join(app.root_path,'resource', 'haarcascade_frontalface_default.xml')
            
        }

    def get_temp_folder(self):
        return self.configs['TEMP_FOLDER']        

    def get_work_folder(self):
        return self.configs['WORK_FOLDER']

    def get_face_classifier_file(self):
        return self.configs['FACE_CLASSIFIER_FILE']

    def get_all_configs(self):
        return self.configs

def init_sysconfig(app):
    config = SysConfig(app)
    return config

def get_sysconfig():
    if 'sysconfig' not in g:
        g.sysconfig = init_sysconfig(current_app)
    return g.sysconfig

def init_config():
    init_sysconfig(current_app)

@click.command('show-configs')
@with_appcontext
def init_config_command():
    click.echo('Initialized config.')
    init_config()                      
    for key, value in get_sysconfig().get_all_configs().items():
        print(key,":",value)

def init_app(app):
    #app.teardown_appcontext(close_db)
    app.cli.add_command(init_config_command)