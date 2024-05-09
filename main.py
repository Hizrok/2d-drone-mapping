#!/usr/bin/python3

##
# @author Jan Kapsa
# @date 9.5.2024
# @file main.py contains main class and the main loop

import argparse
import os
import glob
from lib.strategy import KeypointMapStrategy, PrevImageStrategy
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Stitcher:
	
	def __init__(self, image_directory, strat="KM", limit=None, skip_n=0):
		
		# initialize queue of images
		self.directory = image_directory
		self.queue = []
		self.get_image_sequence()

		# set image limit
		self.limit = limit if limit is not None else len(self.queue)
		
		# skip first n images
		for _ in range(skip_n):
			if len(self.queue):
				self.queue.pop(0)
			else:
				break
		
		self.counter = 1
		
		if strat == "KM":
			self.strategy = KeypointMapStrategy()
		else: 
			self.strategy = PrevImageStrategy()
	
	## @brief starts the stitching process
	def run(self):
		if len(self.queue) == 0:
			return
		
		initial_image_path = self.queue.pop(0)
		self.strategy.initialize(initial_image_path)

		while len(self.queue) and self.counter < self.limit:
			new_image_path = self.queue.pop(0)
			self.strategy.new_image(new_image_path, self.counter)
			self.counter += 1

		cv2.imwrite("./out/map.jpg", self.strategy.mosaic)
		
		bbox_strings = []
		min_y = float('inf')
		max_y = float('-inf')
		
		for bbox in self.strategy.bounding_boxes:

			string = [f"{pt[0]};{pt[1]}" for pt in bbox]
			string = ",".join(string)
			bbox_strings.append(string)
			
			plot_bbox(bbox, 'blue')
			
			min_y_bbox = np.min(bbox[:, 1])
			max_y_bbox = np.max(bbox[:, 1])
			min_y = min(min_y, min_y_bbox)
			max_y = max(max_y, max_y_bbox)
		
		plt.ylim(max_y+50, min_y-50)
		plt.show()
	
	## @brief finds jpg files in a directory and puts them into an image queue ordered alphabeticaly
	def get_image_sequence(self):
		pattern = os.path.join(self.directory, '*.[jJ][pP][gG]')
		image_paths = glob.glob(pattern)
		image_paths.sort(key=lambda item: (len(item), item))
		self.queue = image_paths

## @brief visualization
def plot_bbox(bbox, color='red', style='-'):
	for i in range(len(bbox)):
		current_point = bbox[i]
		next_point = bbox[(i + 1) % len(bbox)]
		plt.plot([current_point[0], next_point[0]], [current_point[1], next_point[1]], style, color=color)
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog='Drone mapping', description='generates a 2D map of the area given a sequence of images')
	
	parser.add_argument("dirpath", help="path to the directory containing images")
	parser.add_argument("-s", "--strat", choices=["KM", "PI"], help="selects the strategy to use - KM for keypoint map and PI for previous image")
	parser.add_argument("-l", "--limit", type=int, help="limits the number of processed images")
	parser.add_argument("-n", "--skip_n", type=int, help="skips n first images in the sequence")

	args = parser.parse_args()

	s = Stitcher(args.dirpath, strat=args.strat, limit=args.limit, skip_n=args.skip_n)
	s.run()