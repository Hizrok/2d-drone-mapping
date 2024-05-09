#!/usr/bin/python3

##
# @author Jan Kapsa
# @date 9.5.2024
# @file extractor.py extracts features and descriptors with SIFT

import cv2

## @class Extractor
# @brief handles extraction of features and descriptors
class Extractor:
	
	def __init__(self, downscale=0, n_features=0):
		
		self.downscale = downscale
		
		if n_features != 0:
			self.extractor = cv2.SIFT_create(n_features)
		else:
			self.extractor = cv2.SIFT_create()

	
	def extract(self, image_path: str):
		img = cv2.imread(image_path)
		
		if self.downscale:
			img = img[::self.downscale,::self.downscale]
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		features, descriptors = self.extractor.detectAndCompute(gray, None)

		return img, features, descriptors
