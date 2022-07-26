import cv2 as cv
from cv2 import Laplacian
import numpy as np

def dec_int(img):
	
	width = img.shape[1]
	height = img.shape[0]
	
	# dividir a imagem em 3 imagens unidimensionais para cada canal RGB,
	# e criar 3 imagens unidimensionais correspondentes a cada canal
	# com metade do tamanho da imagem original

	b, g, r = cv.split(img)
	half_b  = np.zeros((height//2, width//2, 1), dtype=np.uint8)
	half_g  = np.zeros((height//2, width//2, 1), dtype=np.uint8)
	half_r  = np.zeros((height//2, width//2, 1), dtype=np.uint8)

	x, y = 0, 0

	# print("image (width, height): (" + str(width) + ", " + str(height) + ")\n" )
	
	# para cada grupo de 4 pixels da imagem original, 
	# interpolar utilizando o valor do pixel mais proximo
	for row in range(0, (height - 1), 2):
		for column in range(0, (width - 1), 2):
			# print("(x, y) = (" + str(x) + ", " + str(y) + ") ; (row,column) = (" +  str(row) + ", " + str(column) + ")" + "\n")
			half_b[x, y] = b[row, column]
			half_g[x, y] = g[row, column]
			half_r[x, y] = r[row, column]
			y += 1
		y = 0
		x += 1

	#recombinar os canais para formar a imagem final RGB
	return cv.merge((half_b, half_g, half_r))

def edge_improv2(img):
	
	#Aplicar o operador Gaussiano para reduzir o ruido ao aplicar o filtro de agucamento
	img_blurred = cv.GaussianBlur(img, (5,5), 0)
	
	#converter a imagem para escala de cinza
	img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)

	#aplicar o operador laplaciano na imagem
	#utilizar depth da imagem como CV_16S para evitar overflow
	img_lapl = cv.Laplacian(img_gray, ddepth=cv.CV_16S, ksize=1)

	
	#converter para uint8
	img_lapl_uint8 = cv.convertScaleAbs(img_lapl)
	
	b, g, r = cv.split(img)

	#subtrair cada canal da imagem original com a imagem resultante do operador Laplaciano
	b_filtered = cv.subtract(b, img_lapl_uint8)
	g_filtered = cv.subtract(g, img_lapl_uint8)
	r_filtered = cv.subtract(r, img_lapl_uint8)
	
	final_image = cv.merge((b_filtered, g_filtered, r_filtered))
    
	return final_image

	#mostrar imagem
	#cv.imshow('Edge_Sub', final_image)
	#cv.waitKey(0)
			
def prog1():
	img = cv.imread('img/test80.jpg')

	half_width = img.shape[1]//2
	half_height = img.shape[0]//2

	#reduzir e interpolar usando dec_int()
	img_interpolated =  dec_int(img)	
	cv.imwrite("output/q1/prog1-dec-int.jpg", img_interpolated)
	cv.imshow('dec_int()', img_interpolated)
    
    #reduzir e interpolar utilizando interpolacao bicubica
	img_bicubic =  cv.resize(img, [half_width, half_height], cv.INTER_CUBIC)	
	cv.imwrite("output/q1/prog1-bicubic.jpg", img_bicubic)
	cv.imshow('interpolacao bicubica', img_bicubic)

	#utilizar filtro de agucamento em ambas imagens
	img_interpolated_edge = edge_improv2(img_interpolated)
	cv.imwrite("output/q1/prog1-dec-int-edge.jpg", img_interpolated_edge)
	cv.imshow('dec_int() - filtro de agucamento', img_interpolated_edge)

	img_bicubic_edge = edge_improv2(img_bicubic)
	cv.imwrite("output/q1/prog1-bicubic-edge.jpg", img_bicubic_edge)

	cv.imshow('interpolacao bicubica - filtro de agucamento', img_bicubic_edge)

	cv.waitKey(0)
	
prog1()

		
		

	

