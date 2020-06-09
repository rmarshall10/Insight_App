from flask import render_template
from flask import request
from app import app
from app import video_tracker
from flask import send_from_directory
import os
from werkzeug import secure_filename
import posenet
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html", title = 'Home', user = { 'nickname': 'rockstar!' })


@app.route('/upload')
def upload_file():
	return render_template('upload.html')

def allowed_extension(filename):
	if not "." in filename:
		return False
	ext = filename.rsplit(".",1)[1]

	if ext.upper() in app.config['ALLOWED_EXTENSIONS']:
		return True
	else:
		return False

	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploaded_file():
	if request.method == 'POST':
		f = request.files['file']
		# if not allowed_extension(f.filename):
		# 	return redirect(?????)

		f.save(secure_filename(f.filename))
		#return 'file uploaded successfully'
		model_path = "app/static/"
		model_name = "frozen_inference_graph.pb"
		model_text = "graph_text.pbtxt"
		net = video_tracker.loading_model(model_path + model_name, model_path + model_text)
		
		with tf.Session() as sess:
			model_cfg, model_outputs = posenet.load_model(101, sess)
			output_stride = model_cfg['output_stride']

			(bounces, body_part_bounces, body_part_sequence) = video_tracker.run_video(f.filename, net, sess, output_stride, model_outputs)

		return str(bounces) + '\n' + str(body_part_sequence)

		#return render_template('video_output.html')

