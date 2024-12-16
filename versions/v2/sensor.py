#======================================================================================
#=================================================================================== V2
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

#============================================================================== Global variables

# Shared readonly
wtd = Watchdog()
wtd.open()
wtd.timeout = 300#60

try:
	cfg = utils.loadConfig('/home/pi/Desktop/BusSensor/config.json')
except:
	with open(cfg['application']['log-file-path'], 'a') as log:
		log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Error while loading configuration\n")
		exit()		

# Shared
frames = []
notifier = None

# Locks
framesLock = threading.Lock()
notifierLock = threading.Lock()

#============================================================================== Utility functions

# Compute how many seconds to wait
def computeWaitTime(delta: int):
	global cfg
	
	waitTime = -54.00 * delta + cfg['application']['max-wait']
	
	return waitTime

#============================================================================== Thread functions

# Consumer
def consumer():
	# Config
	global cfg

	# Shared variables
	global notifier
	global frames

	# Lock variables
	global framesLock
	global notifierLock
	
	#print('Consumer')
	
	# Local variables
	manager = utils.InfluxManager(cfg)
	model = YOLO(cfg['application']['model-path'])
	countPrev = 0
	
	print('===== Consumer Loaded')
	
	# Consumer loop
	while True:
		# Lock frames
		framesLock.acquire()
		# Copy array locally
		_frames = np.copy(frames)
		frames.clear()
		# Release frames
		framesLock.release()

		# If no frame available
		if len(_frames) == 0:
			# Notify success
			notifierLock.acquire()
			notifier = None
			notifierLock.release()
			
			# Sleep
			#print('##################### C - sleep void')
			time.sleep(cfg['application']['min-wait'])
		else:
			# TODO Inference
			#print('##################### C - inference')
			
			# Predict with the model
			counts = []
			for f in _frames:
				#print('##################### C - inference (frame)')
				results = model(f, classes = [0], conf = cfg['application']['inference-conf'])
				counts.append(len(results[0].boxes.cpu()))

			# Get max count
			count = max(counts)
			# Compute delta
			delta = abs(count - countPrev)
			# Update previous count
			countPrev = count
			
			#print(f'##################### C - Count {count}')
			#print(f'##################### C - Delta {delta}')
			
			#print('##################### C - send')
			# Send data
			manager.sendCount(count)
			manager.sendDelta(delta)

			#print('##################### C - notify')
			# Notify success
			notifierLock.acquire()
			notifier = None
			notifierLock.release()
			
			# Determine the seconds which the sensor should sleep
			waitTime = computeWaitTime(delta)
			if waitTime > cfg['application']['max-wait']:
				waitTime = cfg['application']['max-wait']
			if waitTime < cfg['application']['min-wait']:
				waitTime = cfg['application']['min-wait']
			 
			#print('##################### C - draw')
			"""
			############################################################ START DISPLAY
			if cfg['application']['display-frames']:
				# Draw bboxes on frame
				array = results[-1].boxes.xyxy.cpu()
				for i in array:
					_frames[-1] = cv2.rectangle(_frames[-1],(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,255,),2)
				# Display frame
				cv2.imshow('frame', _frames[-1])
				# Exit on 'q' pressed (inside streaming window)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			############################################################ END DISPLAY
			"""

			#print('##################### C - wait')
			# Sleep
			time.sleep(waitTime)

# Producer
def producer():
	# Config
	global cfg
	
	# Watchdog
	global wtd
	
	# Shared variables
	global notifier
	global frames
	
	# Lock variables
	global framesLock
	global notifierLock

	# Camera
	#print('Producer')
	camera = cv2.VideoCapture(cfg['application']['source'])
	counter = 0
	_buffer = []
	
	print('===== Producer Loaded')
	
	# Watchdog pinging code
	#with Watchdog() as wtd:
	#print(f'WATCHDOG:\n\tTimeout{wtd.timeout}')
	
	# Producer loop
	while camera.isOpened():
		result, frame = camera.read()

		# Ping watchdog on successful read
		if result:
			#print('P - Frame acquired')
			# Watchdog ping
			wtd.keep_alive()
		else:
			print('Unable to read frame')
			
			if wdt.time_left != None and wdt.time_left < 5:
				with open(cfg['application']['log-file-path'], 'a') as log:
					log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())} - Unable to read frames in the last 300 seconds\n')
		
		# Enqueue new frame
		if counter % cfg['application']['frame-to-skip'] == 0 and result:
			_buffer.append(frame)
			# Counter to save frames
		
		counter += 1
		
		#print(f'P - BUFFER = {len(_buffer)}')
		
		if len(_buffer) == 4:
			# Lock frames
			framesLock.acquire()
			# Copy buffer
			frames = _buffer.copy()
			# Unlock frames
			framesLock.release()

			# Flush buffer
			_buffer.clear()
		
		# Check notifier
		# TODO optimize
		#print('P - check for reboot')
		notifierLock.acquire()
		_notifier = notifier
		notifierLock.release()
		
		if _notifier == None:
			_notifier = time.time()
			
			notifierLock.acquire()
			notifier = _notifier
			notifierLock.release()
		
		elif time.time() - _notifier > cfg['application']['consumer-max-downtime']:
			with open(cfg['application']['log-file-path'], 'a') as log:
				log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Consumer not working ({cfg['application']['consumer-max-downtime']} seconds)\n")
			
			# Reboot
			os.system('systemctl reboot')

#============================================================================== Main

def main():
	# Create threads
	pThread = threading.Thread(target=producer)
	cThread = threading.Thread(target=consumer)
	
	# Start threads
	pThread.start()
	cThread.start()

	# Join threads
	pThread.join()
	cThread.join()

#============================================================================== Application

print('===== BUS SENSOR =====')
main()
