#======================================================================================
#=================================================================================== V1
#======================================================================================

import cv2
import threading
import numpy as np
import time
from pywatchdog import Watchdog
import subprocess
from ultralytics import YOLO

#============================================================================== Global variables

# Shared
frames = []
notifier = None

# Locks
framesLock = threading.Lock()
notifierLock = threading.Lock()
readyLock = threading.Lock()

#============================================================================== Utility functions

# Send data to influx
def sendInflux(data):
	pass

def computeWaitTime(delta: int):
	pass

#============================================================================== Thread functions

# Consumer
def consumer():
	# Shared variables
	global notifier
	global frames

	# Lock variables
	global framesLock
	global notifierLock
	global readyLock
	
	# Local variables
	model = YOLO("yolov8n.pt")
	countPrev = 0

	# Wait until the queue is ready
	while readyLock.locked():
		pass
	
	# Consumer loop
	while True:
		# Lock frames
		framesLock.acquire()
		# Copy array locally
		_frames = np.copy(frames)
		# Release frames
		framesLock.release()

		# TODO Inference
		# Predict with the model
		counts = []
		results = model(_frames, classes = [0], conf = 0.4)
		for r in results:
			counts.append(len(r.boxes.cpu()))

		# Get max count
		count = max(counts)

		# Send data
		sendInflux(count)

		# Notify success
		notifierLock.acquire()
		notifier = None
		notifierLock.release()

		# Compute delta
		delta = abs(count - countPrev)
		# Update previous count
		countPrev = count

		# Determine the seconds which the sensor should sleep
		waitTime = computeWaitTime()

		# Sleep
		time.sleep(waitTime)

# Producer
def producer(cameraUri: str | int = 0, consumerMaxWait: int = 300):
	# Shared variables
	global notifier
	global frames
	
	# Lock variables
	global framesLock
	global notifierLock
	global readyLock

	# Camera
	camera = cv2.VideoCapture(cameraUri)
	counter = 0
	
	# Watchdog pinging code
	with Watchdog() as wtd:

		# Producer loop
		while camera.isOpened():
			result, frame = camera.imread()

			# Break on reading error
			if not result:
				break
			
			# Watchdog ping
			wtd.keep_alive()

			# Enqueue new frame
			if counter % 3 == 0:
				# Lock frames
				framesLock.acquire()
				
				frames.append(frame)

				if len(frames) == 5:
					frames.pop(0)

					if readyLock.locked():
						readyLock.release()
				
				# Unlock frames
				framesLock.release()

			# Counter to save frames
			counter += 1
			
			# Check notifier
			# TODO optimize
			notifierLock.acquire()
			if notifier == None:
				notifier = time.time()
				notifierLock.release()
			elif time.time() - notifier > consumerMaxWait:
				notifierLock.release()
				# Reboot
				subprocess.run('systemctl reboot')
				pass
		
		# Will cause watchdog to reboot the system
		exit()

#============================================================================== Main

def main():
	# Lock to stop the consumer
	#global readyLock

	# Lock to block consumer
	#readyLock.acquire() 
	
	# Create threads
	pThread = threading.Thread(producer)
	cThread = threading.Thread(consumer)
	
	# Start threads
	pThread.start()
	cThread.start()

	# Join threads
	pThread.join()
	cThread.join()

#============================================================================== Application

if __name__ == 'main':
	main()
