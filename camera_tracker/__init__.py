import os

from flask import Flask
from . import api

def create_app(config = None):

    app = Flask(__name__, instance_relative_config = True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
    )

    if config is None:
        app.config.from_pyfile('config.py', silent = True)
    else:
        app.config.from_mapping(config)

    
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def home():
        return 'camera_tracker api'

    app.register_blueprint(api.bp)
    
    return app
