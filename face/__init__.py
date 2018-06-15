import os
from flask import Flask, render_template
from . import sysconfig
from . import commands

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 megabytes uploading limit
    app.config.from_envvar('SETTINGS')

    #configs = sysconfig.init_sysconfig(app)
    sysconfig.init_app(app)
    app.app_context().push()
    configs = sysconfig.get_sysconfig()

    commands.init_app(app)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def index():
        
        dirList = os.listdir(configs.get_work_folder())
        faces = [dir for dir in dirList if os.path.isdir(os.path.join(configs.get_work_folder(), dir)) == True]

        data = {
            'faces':faces,
            'configs':configs
            }
        
        return render_template('index.html', data = data)

    from . import mainflow
    app.register_blueprint(mainflow.bp)

    from . import sample
    app.register_blueprint(sample.bp)

    from . import train
    app.register_blueprint(train.bp)

    from . import recognize
    app.register_blueprint(recognize.bp)    
    
    return app
