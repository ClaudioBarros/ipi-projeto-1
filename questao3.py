import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def main():
	img_clown = cv.imread("img/clown.tif", 0)
	img_mandrill = cv.imread("img/mandrill.tif", 0)
	
	img_clown_dft = cv.dft(np.float32(img_clown), flags=cv.DFT_COMPLEX_OUTPUT)	
	img_mandrill_dft = cv.dft(np.float32(img_mandrill), flags=cv.DFT_COMPLEX_OUTPUT)		

	# centralizar
	img_clown_fshift = np.fft.fftshift(img_clown_dft)	
	img_mandrill_fshift = np.fft.fftshift(img_mandrill_dft)	

	img_clown_mag = 20*np.log(cv.magnitude(img_clown_fshift[:,:,0], img_clown_fshift[:,:,1]))
	img_mandrill_mag = 20*np.log(cv.magnitude(img_mandrill_fshift[:,:,0], img_mandrill_fshift[:,:,1]))
		
	plt.subplot(121),plt.imshow(img_clown, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img_clown_mag, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()	

	plt.subplot(121),plt.imshow(img_mandrill, cmap = 'gray')
	plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img_mandrill_mag, cmap = 'gray')
	plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])
	plt.show()	
main()
