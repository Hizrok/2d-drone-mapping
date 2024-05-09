#!/usr/bin/python3

import cv2

##
# @author Jan Kapsa
# @date 9.5.2024
# @file matcher.py matches 2 sets of descriptors with FLANN

## @class Matcher
# @brief matches 2 sets of descriptors and finds good matches using the D. Lowe ratio
class Matcher:
	
	def __init__(self):
		self.matcher = None

		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=100)

		self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
	
	def find_matches(self, prev_descriptors, new_descriptors):
		matches = self.matcher.knnMatch(prev_descriptors, new_descriptors, k=2)
		good_matches = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good_matches.append(m)

		# fail safe
		if len(good_matches) < 4:
			exit("[error] not enough matches found for next iteration, exiting...")

		return good_matches

