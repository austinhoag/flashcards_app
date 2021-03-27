import os,sys
from flask import Flask, session
from app.config import Config
# from flask_sqlalchemy import SQLAlchemy
# import datajoint as dj
# from flask_wtf.csrf import CSRFProtect
# import socket

# dj.config['database.host'] = '127.0.0.1'
# dj.config['database.port'] = 3306
# dj.config['database.user'] = 'ahoag'
# dj.config['database.password'] = 'gaoha'
# if socket.gethostname() == 'braincogs00.pni.princeton.edu':

# dj.config['database.user'] = 'root'
# dj.config['database.password'] = 'tutorial'
# dj.config['database.port'] = 3307

# db_admin = dj.create_virtual_module('admin_demo','ahoag_admin_flask_demo',create_schema=True)

# cel = Celery(__name__,broker='amqp://localhost//',
# 	backend='db+mysql+pymysql://ahoag:p@sswd@localhost:3306/ahoag_celery_test')

def create_app(config_class=Config):
	""" Create the flask app instance"""
	app = Flask(__name__)
	# csrf = CSRFProtect(app)

	app.config.from_object(config_class)
	from app.main.routes import main
	app.register_blueprint(main)

	return app
