import cv2 as cv
from cv2 import Laplacian
from cv2 import imread
import numpy as np

def gamma_correct(img, gamma):
	## [changing-contrast-brightness-gamma-correction]
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

	res = cv.LUT(img, lookUpTable)
	## [changing-contrast-brightness-gamma-correction]

	img_gamma_corrected = cv.hconcat([img, res])
	return img_gamma_corrected
	
def main():
	img_car = imread("img/car.png")
	img_crowd = imread("img/crowd.png")
	img_university = imread("img/university.png")

	img_car = cv.cvtColor(img_car, cv.COLOR_BGR2GRAY)
	img_crowd = cv.cvtColor(img_crowd, cv.COLOR_BGR2GRAY)
	img_university = cv.cvtColor(img_university, cv.COLOR_BGR2GRAY)
	
	### Correcao Gamma ###
	gamma = 2.0

	img_car_gamma = gamma_correct(img_car, gamma)
	cv.imshow("Gamma correction - Car", img_car_gamma)	

	gamma = 0.4
	img_crowd_gamma = gamma_correct(img_crowd, gamma)
	cv.imshow("Gamma correction - Crowd", img_crowd_gamma)	

	#gamma = 1.0
	img_university_gamma = gamma_correct(img_university, gamma)
	cv.imshow("Gamma correction - University", img_university_gamma)	
	
	### Equalizacao de Histograma ###

	img_car_hist = cv.equalizeHist(img_car)
	img_crowd_hist = cv.equalizeHist(img_crowd)
	img_university_hist = cv.equalizeHist(img_university)
	
	cv.imshow("Equalizacao de Histograma - Car", cv.hconcat([img_car, img_car_hist]))
	cv.imshow("Equalizacao de Histograma - Crowd", cv.hconcat([img_crowd, img_crowd_hist]))
	cv.imshow("Equalizacao de Histograma - University", cv.hconcat([img_university, img_university_hist]))

	cv.waitKey(0)

	
main()
