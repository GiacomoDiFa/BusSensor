#======================================================================================
#=================================================================================== V4
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

# Compute how many seconds to wait
def computeWaitTime(delta: int):
	global cfg
	
	waitTime = -54.00 * delta + cfg['application']['max-wait']
	
	return waitTime

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
	
#============================================================================== Application

print('===== BUS SENSOR =====')

# InfluxDB manager
manager = utils.InfluxManager(cfg)

# Inference model
model = YOLO(cfg['application']['model-path'])
countPrev = 0

buffer = []
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
	
	#======================================= Send data
	#"""
	try:
		manager.sendCount(count)
		manager.sendDelta(delta)
	except:
		with open(cfg['application']['log-file-path'], 'a') as log:
			log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Error while sending data\n")
		time.sleep(1800) # Will trigger the watchdog
	#"""
	
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
		
	# Watchdog open
	wtd.close()
	
	# Sleep
	time.sleep(waitTime)
