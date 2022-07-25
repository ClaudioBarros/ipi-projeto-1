import cv2 as cv
import numpy as np

def dec_int(img):
	
	width = img.shape[0]
	height = img.shape[1]
	
	b, g, r = cv.split(img)
	half_b, half_g, half_r = np.zeros((height/2, width/2, 3), np.uint8)

	b_iter = iter(half_b)
	g_iter = iter(half_g)
	r_iter = iter(half_r)
	for row in range(height/2):
		for column in range(width/2):
			
			

		
		

	

