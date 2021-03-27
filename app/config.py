# config.py
""" This file contains the setup for the app,
for both testing and deployment """

import os
from datetime import timedelta

# The default config
class Config(object):
	SECRET_KEY = os.environ.get('SECRET_KEY')

