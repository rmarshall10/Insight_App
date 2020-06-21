from flask import render_template
from flask import request, Response, make_response
from app import app
from app import video_tracker
from flask import send_from_directory
import os
from werkzeug import secure_filename
import posenet
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html", title = 'Home', user = { 'nickname': 'rockstar!' })

@app.route('/')
@app.route('/upload')
def upload_file():
	return render_template('upload.html')

def allowed_extension(filename):
	'''This function can be implemented to check if the uploaded file is a video file'''
	if not "." in filename:
		return False
	ext = filename.rsplit(".",1)[1]

	if ext.upper() in app.config['ALLOWED_EXTENSIONS']:
		return True
	else:
		return False

	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploaded_file():
	'''If a video is uploaded, process the video and display output'''
	if request.method == 'POST':
		f = request.files['file']
		# if not allowed_extension(f.filename):
		# 	return

		# save the file
		f.save(secure_filename(f.filename))
	
		# load the soccer ball detection model
		model_path = "app/static/"
		model_name = "frozen_inference_graph.pb"
		model_text = "graph_text.pbtxt"
		net = video_tracker.loading_model(model_path + model_name, model_path + model_text)
		
		# load pose model
		sess = tf.Session()
		model_cfg, model_outputs = posenet.load_model(101, sess)
		output_stride = model_cfg['output_stride']

		#OPTIONALLY can output the following, then show output as pre-loaded gif
		#(video_bytes, bounces, body_part_bounces, body_part_sequence) = video_tracker.run_video(f.filename, net, sess, output_stride, model_outputs)
		#return Response(video_tracker.display_video(video_bytes), mimetype='multipart/x-mixed-replace; boundary=frame')
		
		# Show output video as it is processed in real time
		return Response(video_tracker.run_video(f.filename, net, sess, output_stride, model_outputs), mimetype='multipart/x-mixed-replace; boundary=frame')
			

def example_file(link):
	'''If user clicks on an example link, process the static video file'''

	if link == 1:
		video_name = "ball_test6.mp4"
	elif link == 2:
		video_name = "ball_test3.mp4"
	else:
		video_name = "ball_test6.mp4"

	#load the soccer ball detection model
	model_path = "app/static/"
	model_name = "frozen_inference_graph.pb"
	model_text = "graph_text.pbtxt"
	filename = model_path + video_name

	net = video_tracker.loading_model(model_path + model_name, model_path + model_text)
	
	#load the pose model
	sess = tf.Session()
	model_cfg, model_outputs = posenet.load_model(101, sess)
	output_stride = model_cfg['output_stride']

	#OPTIONALLY can output the following, then show output as pre-loaded gif
	#(video_bytes, bounces, body_part_bounces, body_part_sequence) = video_tracker.run_video(f.filename, net, sess, output_stride, model_outputs)
	#return Response(video_tracker.display_video(video_bytes), mimetype='multipart/x-mixed-replace; boundary=frame')
	
	return Response(video_tracker.run_video(filename, net, sess, output_stride, model_outputs), mimetype='multipart/x-mixed-replace; boundary=frame')
	


@app.route('/example_1')
def example1():
	return example_file(1)

@app.route('/example_2')
def example2():
	return example_file(2)

@app.route('/example_3')
def example3():
	return example_file(3)

@app.route('/about')
def about():	
	return "Insight Fellows project"

@app.route('/contact')
def contact():
	return "Email: ryanmarshall89@gmail.com"