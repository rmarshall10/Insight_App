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




def check_juggle(cYs):
	'''Checks to see if a juggle has been made by looking at position and direction history'''

	if (cYs[-1] <= cYs[-2]) and (cYs[-2] >= cYs[-3]):
		return True
	else:
		return False



def get_centroid(frame, detection, W, H):
	''''''
	box = detection[0, 0, 0, 3:7] * np.array([W, H, W, H])
	

	(startX, startY, endX, endY) = box.astype("int")
	diameter = ((endX - startX) + (endY - startY)) / 2.0
	#cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	cX = int((startX + endX) / 2.0)
	cY = int((startY + endY) / 2.0)

	return cX, cY, diameter


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
	cYs = [0,0] #list of cY for each frame
	
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
			cX, cY, diameter_temp = get_centroid(frame, detection, W, H)
			if first_detection:
				diameter = diameter_temp
				first_detection = False
			if  0.75 * diameter < diameter_temp < 1.25 * diameter:
				cv2.circle(frame, (cX, cY), int(W * 0.03), (0, 255, 0), -1)
				cYs.append(cY)
				if check_juggle(cYs):
					bounces += 1
					#pose detect


			cv2.putText(frame, str(bounces), (int(W * 0.86), int(W * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, int(W * 0.007), (0, 255, 0), 2)

		
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
# net = loading_model('frozen_inference_graph_sc_ball3.pb', 'graph_sc2.pbtxt')
# start_time = time.time()
# bounces = run_video(path, net)
# print(f"You got {bounces} juggles! Try to beat it next time!")

# print("--- %s seconds ---" % (time.time() - start_time))