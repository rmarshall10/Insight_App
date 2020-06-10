from imutils.video import FileVideoStream
import numpy as np
import imutils
import time
import cv2
import posenet
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def loading_model(graph, text):
	'''load the Tensorflow model graph and text files'''
	
	net = cv2.dnn.readNetFromTensorflow(graph, text)

	return net


def process_frame(frame, net, W, H):
	'''Runs the current frame through the detection model'''

	
	blob = cv2.dnn.blobFromImage(frame, size = (W, H), swapRB=True)#,
	#blobFromImage can perform mean subtraction, scaling, and optionally channel swapping
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
	'''Determine the centroid of the detected ball by looking at bounding box'''
	box = detection[0, 0, 0, 3:7] * np.array([W, H, W, H])
	

	(startX, startY, endX, endY) = box.astype("int")
	diameter = ((endX - startX) + (endY - startY)) / 2.0
	#cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	cX = int((startX + endX) / 2.0)
	cY = int((startY + endY) / 2.0)

	return cX, cY, diameter


def get_pose(frame, sess, output_stride, model_outputs, scale_factor = 1):
	'''Passes frame through Posenet to determine to coordinates of key points of body'''
	
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

	draw_image = posenet.draw_skel_and_kp(
		display_image, pose_scores, keypoint_scores, keypoint_coords,
		min_pose_score=0.15, min_part_score=0.05)
	
	return (draw_image, keypoint_scores[0], keypoint_coords[0])


def get_closest_body_part(k_scores, k_coords, cX, cY):
	'''Determine which body part is closest to bell centroid (but not above)'''
		# PART_NAMES = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
 #    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
 #    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]


	distances = []
	for part in [0, 11, 12, 15, 16]:
		if cY > k_coords[part,0]:
			distance = 100000
		else:
			distance = (cX - k_coords[part,1])**2 + (cY - k_coords[part,0])**2
		distances.append(distance)

	#need to add some max distance apart, so it detects if it's ground or somthing that's not the person.
	#print(distances)
	body_part = distances.index(min(distances))
	# 0 = head, 1 = left thigh, 2 = right thigh, 3 = left foot, 4 = right foot
	return body_part


def run_video(path, net, sess, output_stride, model_outputs):
	'''Runs the uploaded video through the juggle counter algorithm
	'''
	cv2.setUseOptimized(True)

	body_part_key = {0:"Head", 1:"Left Thigh", 2:"Right Thigh", 3:"Left Foot", 4:"Right Foot"}
	body_part_bounces = {"Head":0, "Left Thigh":0, "Right Thigh":0, "Left Foot":0, "Right Foot":0}
	body_part_sequence = []
	confidence = 0.0
	(W, H) = (300, 300)
	#direction = 1
	bounces = 0
	#prev_cY = 0
	frame_num = 0
	last_juggle_frame = -4
	first_detection = True
	cYs = [0,0] #list of cY for each frame
	cXs = [0,0]
	skipped_frames = 100	

	print("Processing video...")
	vs = FileVideoStream(path).start()

	while vs.more():
		frames_per_read = 3
		for i in range(frames_per_read):
			if vs.more():
				frame = vs.read()
				frame_num += 1
		try:
			frame = cv2.resize(frame, (W, H))
		except cv2.error as e:
			print("...end of video")
			break
		
		detection = process_frame(frame, net, W, H)
		################
		skip_frame = True
		for bbox in detection[0, 0, :]:
			cX, cY, diameter_temp = get_centroid(frame, detection, W, H)
			if first_detection:
				diameter = diameter_temp
				first_detection = False
			if (0.7 * diameter < diameter_temp < 2 * diameter) and (((cXs[-1] - cX)**2 + (cYs[-1] - cY)**2) < skipped_frames * 4 * (diameter**2)):
				skip_frame = False
				skipped_frames = 1
				break
		if not skip_frame:
			cv2.circle(frame, (cX, cY), int(W * 0.03), (0, 255, 0), -1)
			cXs.append(cX)
			cYs.append(cY)
			if (frame_num - last_juggle_frame > 8) and check_bounce(cYs):
				
				last_juggle_frame = frame_num
				#pose detect
				(frame2, k_scores, k_coords) = get_pose(frame, sess, output_stride, model_outputs)
				#print(k_scores)
				#print(k_coords)
				body_part = get_closest_body_part(k_scores, k_coords, cX, cY)
				print(body_part_key[body_part])
				#print(body_part)
				body_part_sequence.append(body_part)
				body_part_bounces[body_part_key[body_part]] += 1
				bounces += 1
		else:
			skipped_frames += 100
			print('SKIPPED!!!')


		# if detection[0, 0, 0, 2] > confidence:
		# 	cX, cY, diameter_temp = get_centroid(frame, detection, W, H)
		# 	if first_detection:
		# 		diameter = diameter_temp
		# 		first_detection = False
		# 	if  0.75 * diameter < diameter_temp < 1.25 * diameter:
		# 		cv2.circle(frame, (cX, cY), int(W * 0.03), (0, 255, 0), -1)
		# 		cYs.append(cY)
		# 		if check_bounce(cYs):
		# 			bounces += 1
		# 			#pose detect: input frame, output node coordinates
		# 			(frame2, k_scores, k_coords) = get_pose(frame, sess, output_stride, model_outputs)
		# 			#(frame2, k_scores, k_coords) = get_pose(frame)
		# 			#print(k_scores)
		# 			#print(k_coords)
		# 			body_part = get_closest_body_part(k_scores, k_coords, cX, cY)
		# 			#print(body_part)
		# 			body_part_sequence.append(body_part)
		# 			body_part_bounces[body_part_key[body_part]] += 1

					#add body_part to counts matrix

		frame = cv2.resize(frame, (600, 600))
		cv2.putText(frame, str(bounces), (int(2 * W * 0.7), int(W * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.01), (0, 255, 0), 2)
		# cv2.putText(frame, "H:  " + str(body_part_bounces["Head"]), (int(W * 0.66), int(W * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.004), (0, 255, 0), 2)
		# cv2.putText(frame, "LT: " + str(body_part_bounces["Left Thigh"]), (int(W * 0.66), int(W * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.004), (0, 255, 0), 2)
		# cv2.putText(frame, "RT: " + str(body_part_bounces["Right Thigh"]), (int(W * 0.66), int(W * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.004), (0, 255, 0), 2)
		# cv2.putText(frame, "LF: " + str(body_part_bounces["Left Foot"]), (int(W * 0.66), int(W * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.004), (0, 255, 0), 2)
		# cv2.putText(frame, "RF: " + str(body_part_bounces["Right Foot"]), (int(W * 0.66), int(W * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.004), (0, 255, 0), 2)
		
		cv2.putText(frame, "Head: " + str(body_part_bounces["Head"]), (int(2 * W * 0.76), int(W * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.0035), (0, 255, 0), 2)
		cv2.putText(frame, "Thigh: " + str(body_part_bounces["Left Thigh"]+body_part_bounces["Right Thigh"]), (int(2 * W * 0.76), int(W * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.0035), (0, 255, 0), 2)
		cv2.putText(frame, "Foot: " + str(body_part_bounces["Left Foot"]+body_part_bounces["Right Foot"]), (int(2 * W * 0.76), int(W * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.0035), (0, 255, 0), 2)


		
		ret, img = cv2.imencode(".jpg", frame)
		#video_bytes.append(img.tobytes())
		img = img.tobytes()
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
		#yield (,bounces, body_part_bounces, body_part_sequence)
		
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

def display_video(video_bytes):
	''''''
	for frame in video_bytes:
		time.sleep(.07)
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
##############################################



# cv2.setUseOptimized(True)
# path = 'ball_test3.mp4'
# net = loading_model('frozen_inference_graph_sc_ball3.pb', 'graph_sc2.pbtxt')
# start_time = time.time()
# bounces = run_video(path, net)
# print(f"You got {bounces} juggles! Try to beat it next time!")

# print("--- %s seconds ---" % (time.time() - start_time))