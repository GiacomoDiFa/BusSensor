#============================================================================== Imports
import cv2
import time
import utils

#============================================================================== Global variables

cfg = utils.loadConfig('/home/pi/Desktop/BusSensor/config.json')
			
# Counter to save frames
counter = 0

print('Started')
while True:
	# Camera
	camera = cv2.VideoCapture(cfg['application']['source'])
	result, frame = camera.read()

	# Ping watchdog on successful read
	if not result:
		break
	
	print('Frame acquired')
	
	# Save frame
	cv2.imwrite(f'{time.time()}.jpg', frame)
	counter += 1
	
	if counter == 2:
		break
	
	############################################################ START DISPLAY
	# Display frame
	cv2.imshow('Stream', frame)
	# Exit on 'q' pressed (inside streaming window)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	############################################################ END DISPLAY

	time.sleep(15)

camera.release()
