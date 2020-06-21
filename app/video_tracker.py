from imutils.video import FileVideoStream
import numpy as np
import imutils
import time
import cv2
import posenet
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def loading_model(graph, text):
	'''load the soccer ball detection Tensorflow model graph and text files into OpenCV net'''
	
	net = cv2.dnn.readNetFromTensorflow(graph, text)

	return net


def initialize_constants():
	'''Initialize all constants, keys, and counters'''

	body_part_key = {0:"Head", 1:"Left Thigh", 2:"Right Thigh", 3:"Left Foot", 4:"Right Foot"}
	body_part_bounces = {"Head":0, "Left Thigh":0, "Right Thigh":0, "Left Foot":0, "Right Foot":0}
	body_part_key_simple = {0:"Head", 1:"Thigh", 2:"Thigh", 3:"Foot", 4:"Foot", 5:"Ground"}
	body_part_sequence = []
	confidence = 0.0
	(W, H) = (300, 300)
	bounces = 0
	frame_num = 0
	last_juggle_frame = -4
	first_detection = True
	cYs = [0,0] #list of cY for each frame
	cXs = [0,0] #list of cX for each frame
	skipped_frames = 100
	ground = False	


	return (body_part_key, body_part_bounces, body_part_key_simple, body_part_sequence, confidence, W, H, bounces, frame_num, last_juggle_frame, first_detection, cYs, cXs, skipped_frames, ground)


def read_frame(video, frame_num):
	'''Read the next frame of the video'''

	frames_per_read = 2
	for i in range(frames_per_read):
		if video.more():
			frame = video.read()
			frame_num += 1
	return (frame, frame_num)


def process_frame(frame, net, W, H):
	'''Runs the current frame through the detection model'''

	
	blob = cv2.dnn.blobFromImage(frame, size = (W, H), swapRB=True)
	#blobFromImage can perform mean subtraction, scaling, and channel swapping
	net.setInput(blob)
	detection = net.forward()

	return detection



def check_bounce(cYs):
	'''Checks to see if a juggle has been made by looking at position and direction history'''

	if (cYs[-1] <= cYs[-2]) and (cYs[-2] >= cYs[-3]):
		return True
	else:
		return False


def get_centroid(frame, detection, W, H):
	'''Determine the centroid and diameter of the detected ball by looking at bounding box'''
	box = detection[0, 0, 0, 3:7] * np.array([W, H, W, H])
	
	(startX, startY, endX, endY) = box.astype("int")
	diameter = ((endX - startX) + (endY - startY)) / 2.0
	#cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	cX = int((startX + endX) / 2.0)
	cY = int((startY + endY) / 2.0)

	return cX, cY, diameter


def get_pose(frame, sess, output_stride, model_outputs, scale_factor = 1):
	'''Passes frame through Posenet to determine the coordinates of key points of body'''
	
	input_image, display_image, output_scale = posenet.utils._process_input(frame, scale_factor, output_stride)

	heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
		model_outputs,
		feed_dict={'image:0': input_image}
	)

	pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
		heatmaps_result.squeeze(axis=0),
		offsets_result.squeeze(axis=0),
		displacement_fwd_result.squeeze(axis=0),
		displacement_bwd_result.squeeze(axis=0),
		output_stride=output_stride,
		max_pose_detections=1,
		min_pose_score=0.15)

	keypoint_coords *= output_scale

	#drawing nodes and lines on frame
	draw_image = posenet.draw_skel_and_kp(
		display_image, pose_scores, keypoint_scores, keypoint_coords,
		min_pose_score=0.15, min_part_score=0.05)
	
	return (draw_image, keypoint_scores[0], keypoint_coords[0])


def get_closest_body_part(k_scores, k_coords, cX, cY):
	'''Determine which body part is closest to bell centroid (but not above)'''

	# PART_NAMES = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
 	#    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
 	#    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]
	
	#approximate length of thigh. This will be used to help determine if the touch is not part of the bady and instead is the ground or another object.
	knee_distance = ((k_coords[11, 1] - k_coords[13, 1])**2 + (k_coords[11, 0] - k_coords[13, 0])**2)**0.5
	
	#choosing hips instead of knees works better for detecting thigh vs foot
	parts = [0, 11, 12, 15, 16]
	distances = []

	#loop over five body parts that can make a touch
	for part in parts:
		#if the body part is above the ball, then penalize it so it cannot be chosen
		if cY > k_coords[part,0]:
			distance = 100000
		#distance from body part to ball
		else:
			distance = (cX - k_coords[part, 1])**2 + (cY - k_coords[part,0])**2
	
		distances.append(distance)

	body_part = distances.index(min(distances))

	#if detection is too far away from body, or if all body parts are penalized, then the ball hit something else, like the ground
	if abs(cX - k_coords[parts[body_part], 1]) > knee_distance * 2 or sum(distances) == 5 * 100000:
		return 5
	# 0 = head, 1 = left thigh, 2 = right thigh, 3 = left foot, 4 = right foot, 5 = ground
	return body_part


def run_video(path, net, sess, output_stride, model_outputs):
	'''Runs the uploaded video through the juggle counter algorithm
	'''
	
	#initialize constants and counters
	cv2.setUseOptimized(True)
	(body_part_key, body_part_bounces, body_part_key_simple, body_part_sequence, confidence, W, H, bounces, frame_num, last_juggle_frame, first_detection, cYs, cXs, skipped_frames, ground) = initialize_constants()
	
	# start the video stream
	print("Processing video...")
	vs = FileVideoStream(path).start()

	while vs.more() and not ground:

		(frame, frame_num) = read_frame(vs, frame_num)

		#resize the frame to fit the 300x300 mobilenet size
		try:
			frame = cv2.resize(frame, (W, H))
		except cv2.error as e:
			print("...end of video")
			break
		
		# pass the frame through the model and output detection
		detection = process_frame(frame, net, W, H)

		# get the centroid coordinate and size of ball detected
		cX, cY, diameter_temp = get_centroid(frame, detection, W, H)
		
		# Initialize ball: Determine the ball size from the detection on the first frame (most likely to be clear/stationary ball)
		if first_detection:
			diameter = diameter_temp
			first_detection = False

		# If the detection is not the same size as the known ball, skip it. Also only allow detections within a realistic distance from the previous detection (two diameters of the ball * # of consecutive skipped frames) 	
		skip_frame = True
		
		if (0.7 * diameter < diameter_temp < 2 * diameter) and (((cXs[-1] - cX)**2 + (cYs[-1] - cY)**2)**0.5 < skipped_frames * 2 * diameter):
			skip_frame = False
			skipped_frames = 1
			
		#if we've determined that there is a valide detection, show the detection of ball and add centroid to lists
		if not skip_frame:
			cv2.circle(frame, (cX, cY), int(W * 0.03), (0, 255, 0), -1)
			cXs.append(cX)
			cYs.append(cY)

			#check if there was a juggle, and make sure it has been long enough since last juggle (this is a dirty fix for some false detections)
			if (frame_num - last_juggle_frame > 8) and check_bounce(cYs):
				
				last_juggle_frame = frame_num
				
				#pose detect (pose detect first output is a frame with the pose drawn on. If we don't want to show the pose)
				(_, k_scores, k_coords) = get_pose(frame, sess, output_stride, model_outputs)
				
				# determine which body part is closes to the ball
				body_part = get_closest_body_part(k_scores, k_coords, cXs[-2], cYs[-2])
				
				# add one juggle to the correct body part and overall score
				if body_part == 5:
					ground = True
				else:
					body_part_bounces[body_part_key[body_part]] += 1
					bounces += 1
				body_part_sequence.append(body_part)
		
		#if there is no valid detection in the frame, increase skipped_frames counter
		else:
			skipped_frames += 0.7
			

		#print scores on the frame
		frame = cv2.resize(frame, (600, 600))
		cv2.putText(frame, str(bounces), (int(2 * W * 0.76), int(W * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.01), (0, 255, 0), 2)

		
		# Determining right vs left is not accurate enough, so don't differentiate between them
		cv2.putText(frame, "Head: " + str(body_part_bounces["Head"]), (int(2 * W * 0.76), int(W * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.0035), (0, 255, 0), 2)
		cv2.putText(frame, "Thigh: " + str(body_part_bounces["Left Thigh"]+body_part_bounces["Right Thigh"]), (int(2 * W * 0.76), int(W * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.0035), (0, 255, 0), 2)
		cv2.putText(frame, "Foot:  " + str(body_part_bounces["Left Foot"]+body_part_bounces["Right Foot"]), (int(2 * W * 0.76), int(W * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.0035), (0, 255, 0), 2)

		#Print body part of previous juggle, if a juggle has been made
		try:
			cv2.putText(frame, body_part_key_simple[body_part_sequence[-1]], (int(2 * W * 0.05), int(W * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.007), (0, 255, 0), 2)
		except:
			pass

			
		# Show the frame
		ret, img = cv2.imencode(".jpg", frame)
		img = img.tobytes()
		#video_bytes.append(img)
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
		#yield (bounces, body_part_bounces, body_part_sequence)
		

		#if run in terminal instead of app, can have OpenCV show image
		#cv2.imshow("Frame", frame)
		#key = cv2.waitKey(1) & 0xFF

		# Only for debuggin:
		# if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	break
	cv2.destroyAllWindows()
	vs.stop()

	#eturn (video_bytes, bounces, body_part_bounces, body_part_sequence)
	return (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


# Optional function to display the frames from a list like a video in real time
# def display_video(video_bytes):
# 	'''Takes a list of image bytes and yields them consecutively to play in browser like a video'''
# 	for frame in video_bytes:
# 		time.sleep(.07)
# 		yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
