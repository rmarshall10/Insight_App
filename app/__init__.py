from flask import Flask
app = Flask(__name__)

#This is wrong, needs FULL path. Right now will save in app folder
#app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['MAX_CONTENT_PATH'] = 10 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {"MP4"}
from app import action