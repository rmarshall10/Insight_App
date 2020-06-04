from imutils.video import FileVideoStream
import numpy as np
import imutils
import time
import cv2



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


def check_detection(frame_num, prev_frame_num, frame, detection, W, H, bounces, direction, prev_cY):
	'''For the most likely detected soccer ball, creates the bounding box and finds centroid.
	Then, checks to determine if a juggle happened since the last detected soccer ball'''

	box = detection[0, 0, 0, 3:7] * np.array([W, H, W, H])
	

	(startX, startY, endX, endY) = box.astype("int")
	diameter = ((endX - startX) + (endY - startY)) / 2.0
	#cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	cX = int((startX + endX) / 2.0)
	cY = int((startY + endY) / 2.0)

	(bounces, direction, prev_cY, prev_frame_num) = check_juggle(frame_num, prev_frame_num, bounces, direction, cY, prev_cY)
		
	return (cX, cY, bounces, direction, prev_cY, prev_frame_num, diameter)


def check_juggle(frame_num, prev_frame_num, bounces, direction, cY, prev_cY):
	'''Checks to see if ball changes direction from going down to up
	to count as a juggle'''

	if (direction == -1) and (cY > prev_cY):		
		direction = 1
	if (direction == 1) and (cY < prev_cY) and (frame_num - prev_frame_num > 5):
		direction = -1
		bounces += 1
		prev_frame_num = frame_num

		#Pose detection goes here...

	prev_cY = cY

	return (bounces, direction, prev_cY, prev_frame_num)


def run_video(path, net):
	'''Runs the uploaded video through the juggle counter algorithm
	'''
	cv2.setUseOptimized(True)

	confidence = 0.45
	(W, H) = (200, 200)
	direction = 1
	bounces = 0
	prev_cY = 0
	frame_num = 0
	prev_frame_num = -7
	first_detection = True
	
	print("Processing video...")
	vs = FileVideoStream(path).start()

	while vs.more():
		frames_per_read = 2
		for i in range(frames_per_read):
			frame = vs.read()
			frame_num += 1
		try:
			frame = cv2.resize(frame, (W, H))
		except cv2.error as e:
			print("...end of video")
			break
		
		detection = process_frame(frame, net, W, H)
		if detection[0, 0, 0, 2] > confidence:
			(cX, cY, bounces, direction, prev_cY, prev_frame_num, diameter_temp) = check_detection(frame_num, prev_frame_num, frame, detection, W, H, bounces, direction, prev_cY)
			if first_detection:
				diameter = diameter_temp
				first_detection = False
			cv2.putText(frame, str(bounces), (int(W * 0.86), int(W * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.007), (0, 255, 0), 2)
			if  0.9 * diameter < diameter_temp < 1.1 * diameter:
				cv2.circle(frame, (cX, cY), int(W * 0.03), (0, 255, 0), -1)
		
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# Only for debuggin:
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	cv2.destroyAllWindows()
	vs.stop()

	return bounces


##############################################



# cv2.setUseOptimized(True)
# path = 'ball_test3.mp4'
# #path = "juggle_clip2.mp4"
# net = loading_model('frozen_inference_graph_sc_ball3.pb', 'graph_sc2.pbtxt')
# start_time = time.time()
# bounces = run_video(path, net)
# print(f"You got {bounces} juggles! Try to beat it next time!")

# print("--- %s seconds ---" % (time.time() - start_time))