#======================================================================================
#=================================================================================== V5
#======================================================================================

#============================================================================== Imports
import cv2
import threading
import numpy as np
import time
from pywatchdog import Watchdog
import os
from ultralytics import YOLO
import utils
import sys
import tflite_runtime.interpreter as tflite
import requests
import datetime

#============================================================================== Global variables
# Watchdog
wtd = Watchdog()
wtd.open()
wtd.timeout = 300

# Shared readonly
try:
	cfg = utils.loadConfig('/home/pi/Desktop/BusSensor/config.json')
except:
	print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Error while loading configuration\n")
	#exit()
	time.sleep(1800) # Will trigger the watchdog

# Use display
useDisplay = False
if len(sys.argv) > 1 and sys.argv[1] == '--display':
	useDisplay = True

#============================================================================== Utility functions

def computeMQ():
	global cfg
	
	x0 = 0
	x1 = 10
	
	y0 = cfg['application']['max-wait']
	y1 = cfg['application']['min-wait']
	
	m = (y0 - y1) / (x0 - x1)
	q = cfg['application']['max-wait']
	
	return (m, q)

# Compute how many seconds to wait
def computeWaitTime(delta: int):
	m,q = computeMQ()

	waitTime = m * delta + q
	
	return waitTime

# Acquire frames
def acquireFrame():
	global cfg
	
	# Camera
	camera = cv2.VideoCapture(cfg['application']['source'])
	
	# Read frame
	result, frame = camera.read()
	
	# Close camera
	camera.release()
	
	if not result:
		raise Exception("Unable to read frame")
	else:
		return frame

# Do the prediction of the number of the people
def predict(interpreter, input_details, output_details, input_data):
	input_data = np.reshape(input_data,(1,4,1))
	input_data = input_data.astype(np.float32)
	interpreter.set_tensor(input_details[0]['index'],input_data)
	interpreter.invoke()
	prediction = interpreter.get_tensor(output_details[0]['index'])
	input_data = np.reshape(input_data,(1,4))
	return prediction, input_data
	
#============================================================================== Application

print('===== BUS SENSOR =====')


m,q = computeMQ()
print(f'M: {m}\t\tQ: {q}')

# InfluxDB manager
manager = utils.InfluxManager(cfg)

# Inference model
model = YOLO(cfg['application']['model-path'])
countPrev = 0

# Bus counter predictor model
interpreter = tflite.Interpreter(model_path="/home/pi/Desktop/BusSensor/bus_count_predictor.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

buffer = []
buffer_input_predict = None
prediction = 0
cansendprediction = False

wtd.close()

# Application loop
while True:
	# Watchdog open
	wtd.open()
	wtd.timeout = 300
	
	#======================================= Camera
	buffer.clear()
	try:
		##### Frame 1
		frame = acquireFrame()
		wtd.keep_alive()
		buffer.append(frame)
		time.sleep(0.5)
		
		##### Frame 2
		frame = acquireFrame()
		wtd.keep_alive()
		buffer.append(frame)
		time.sleep(0.5)
		
		##### Frame 3
		frame = acquireFrame()
		wtd.keep_alive()
		buffer.append(frame)
		time.sleep(0.5)
		
		##### Frame 4
		frame = acquireFrame()
		wtd.keep_alive()
		buffer.append(frame)
	except:
		with open(cfg['application']['log-file-path'], 'a') as log:
			log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Error while reading frames\n")
		time.sleep(1800) # Will trigger the watchdog
	
	#======================================= Inference
	counts = []
	for f in buffer:
		results = model(f, classes = [0], conf = cfg['application']['inference-conf'])
		wtd.keep_alive()
		counts.append(len(results[0].boxes.cpu()))

	# Get max count
	count = max(counts)
	# Compute delta
	delta = abs(count - countPrev)
	# Update previous count
	countPrev = count
	# Check if the buffer input predict is less than buffer input len predict 
	if buffer_input_predict is None:
		# Build the buffer for the first buffer input predict len times
		buffer_input_predict = np.full((4,1),count)
	#do the prediction
	prediction, buffer_input_predict = predict(interpreter,input_details,output_details,buffer_input_predict)
	rounded_prediction = round(prediction[0][0])
	#======================================= Send data
	try:
		manager.sendCount(count)
		manager.sendDelta(delta)
		if cansendprediction:
			manager.sendPrediction(rounded_prediction)
			url = cfg['influx']['urlAdriabus']
			codLocalita = cfg['influx']['location']
			datehourevent = str(datetime.datetime.now())
			resp = requests.post(url,json={'CodiceLocalita':codLocalita,'NumPersone':int(count),'NumPersonePrediction':int(rounded_prediction),'DataOraEvento':datehourevent,'Note':'Prova'})
			#print(resp.text)
		cansendprediction = True
	except:
		with open(cfg['application']['log-file-path'], 'a') as log:
			log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Error while sending data\n")
		time.sleep(1800) # Will trigger the watchdog
	
	
	# Update input of the new prediction
	buffer_input_predict = buffer_input_predict[0].tolist()
	buffer_input_predict.pop(0)
	buffer_input_predict.append(count)
	buffer_input_predict = np.array([buffer_input_predict])


	# Determine the seconds which the sensor should sleep
	waitTime = computeWaitTime(delta)
	if waitTime > cfg['application']['max-wait']:
		waitTime = cfg['application']['max-wait']
	if waitTime < cfg['application']['min-wait']:
		waitTime = cfg['application']['min-wait']
	
	# Debug prints
	print(f'Bus Stop Count: {count}')
	print(f'Bus Stop Delta: {delta}')
	print(f'Wait Time: {waitTime} seconds')
	print("prediction array: ", prediction)
	print("prediction value: ", rounded_prediction)
	print("buffer input: ", buffer_input_predict)
	
	# Last inference draw on frame
	array = results[-1].boxes.xyxy.cpu()
	drawnFrame = buffer[-1].copy()
	for i in array:
		drawnFrame = cv2.rectangle(drawnFrame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0,255,) ,2)
	
	if cfg['application']['save-last-frame']:
		cv2.imwrite(cfg['application']['last-frame-path'], drawnFrame)
	
	############################################################ START DISPLAY
	if useDisplay:
		# Display frame
		cv2.imshow('frame', drawnFrame)
		# Exit on 'q' pressed (inside streaming window)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	############################################################ END DISPLAY
		
	# Watchdog closed
	wtd.close()
	
	# Sleep
	time.sleep(waitTime)
