#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import os
import glob

# SIFT object
sift = cv2.SIFT_create()

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

class Keypoint:
	def __init__(self, index, point, descriptor):
		self.index = index
		self.point = point
		self.descriptor = descriptor
		self.score = 0.5

class Map:
	def __init__(self):
		self.keypoints: list[Keypoint] = []

	def add_keypoints(self, new_keypoints: list[Keypoint]):
		self.keypoints += new_keypoints

	def get_keypoints(self, polygon: Polygon):
		keypoints: list[Keypoint] = []

		for keypoint in self.keypoints:
			point = Point(keypoint.point)
			if polygon.contains(point):
				keypoints.append(keypoint)

		return keypoints
	
class Generator:
	def __init__(self, queue, limit=None, skip_n=0):
		self.map = Map()
		
		self.queue = queue
		self.limit = limit if limit is not None else len(queue)
		self.counter = 1
		for _ in range(skip_n):
			if len(self.queue):
				self.queue.pop(0)
			else:
				break

		self.incremental_bboxes = []
		self.map_bboxes = []

	def run(self):
		
		if len(self.queue) == 0:
			return

		prev_features, prev_descriptors, prev_homography, prev_bbox = self.initialize()

		self.incremental_bboxes.append(prev_bbox)
		self.map_bboxes.append(prev_bbox)

		while len(self.queue) and self.counter < self.limit:
			print(self.counter)
			img_path = self.queue.pop(0)
			prev_features, prev_descriptors, prev_homography = self.incremental_new_image(img_path, prev_features, prev_descriptors, prev_homography)
			prev_bbox = self.map_new_image(img_path, self.counter, prev_bbox)
			self.counter += 1
			self.plot_bboxes(self.incremental_bboxes, self.map_bboxes)
			self.plot_map()

		pass

	def initialize(self):
		img_path = self.queue.pop(0)
		img = cv2.imread(img_path)[::4,::4]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		features, descriptors = sift.detectAndCompute(gray, None)
		homography = [[1,0,0], [0,1,0], [0,0,1]]

		h, w = img.shape[:2]
		bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

		keypoints = [Keypoint(0, features[i].pt, descriptors[i]) for i in range(len(features))]

		self.map.add_keypoints(keypoints)

		return features, descriptors, homography, bbox

	def incremental_new_image(self, img_path, prev_features, prev_descriptors, prev_homography):
		img = cv2.imread(img_path)[::4,::4]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		features, descriptors = sift.detectAndCompute(gray, None)

		h, w = img.shape[:2]
		bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

		matches = self.find_matches(prev_descriptors, descriptors)

		src_pts = [features[m.trainIdx].pt for m in matches]
		dst_pts = [prev_features[m.queryIdx].pt for m in matches]

		homography = self.find_homography(src_pts, dst_pts)
		homography = np.matmul(prev_homography, homography)

		transformed_bbox = cv2.perspectiveTransform(bbox.reshape(-1, 1, 2), homography)
		transformed_bbox = np.squeeze(transformed_bbox)
		self.incremental_bboxes.append(transformed_bbox)
		
		return features, descriptors, homography
	
	def map_new_image(self, img_path, index, prev_bbox):
		img = cv2.imread(img_path)[::4,::4]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		features, descriptors = sift.detectAndCompute(gray, None)

		h, w = img.shape[:2]
		bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

		# enlarged_bbox = prediction window
		enlarged_bbox = []
		polygon = Polygon(prev_bbox)
		centroid = np.array([polygon.centroid.x, polygon.centroid.y])
		
		for point in prev_bbox:
			vector = np.array([point[0] - centroid[0], point[1] - centroid[1]])
			norm = np.linalg.norm(vector)
			norm_vector = vector / norm
			dist = np.linalg.norm(point - centroid)
			enlarged_bbox.append(centroid + (norm_vector * dist * 1.5))
		enlarged_bbox = np.array(enlarged_bbox, dtype=np.float32)

		self.plot_prediction(prev_bbox, enlarged_bbox)
		
		keypoints: list[Keypoint] = self.map.get_keypoints(Polygon(enlarged_bbox))
		map_points = [keypoints[i].point for i in range(len(keypoints))]
		map_descriptors = np.float32([keypoints[i].descriptor for i in range(len(keypoints))])

		matches = self.find_matches(map_descriptors, descriptors)

		src_pts = [features[m.trainIdx].pt for m in matches]
		dst_pts = [map_points[m.queryIdx] for m in matches]

		homography = self.find_homography(src_pts, dst_pts)

		transformed_bbox = cv2.perspectiveTransform(bbox.reshape(-1, 1, 2), homography)
		transformed_bbox = np.squeeze(transformed_bbox)
		self.map_bboxes.append(transformed_bbox)

		# TODO: only add points that are not matches
		new_points = np.array([kp.pt for kp in features])
		new_points = cv2.perspectiveTransform(new_points.reshape(-1, 1, 2), homography)
		new_points = np.squeeze(new_points)

		self.map.add_keypoints([Keypoint(index, new_points[i], descriptors[i]) for i in range(len(descriptors))])

		return transformed_bbox

	def plot_bboxes(self, bboxes1, bboxes2):
		min_y = float('inf')
		max_y = float('-inf')
		for i in range(len(bboxes1)):
			bbox1 = bboxes1[i]
			bbox2 = bboxes2[i]
			
			plot_bbox(bboxes1[i], 'red')
			plot_bbox(bboxes2[i], 'blue')
			
			min_y_bbox1 = np.min(bbox1[:, 1])
			max_y_bbox1 = np.max(bbox1[:, 1])
			min_y_bbox2 = np.min(bbox2[:, 1])
			max_y_bbox2 = np.max(bbox2[:, 1])
			
			min_y = min(min_y, min_y_bbox1, min_y_bbox2)
			max_y = max(max_y, max_y_bbox1, max_y_bbox2)
		plt.ylim(max_y+50, min_y-50)
		plt.show()

	def plot_prediction(self, prev_bbox, enlarged_bbox):
		min_y = np.min(enlarged_bbox[:, 1])
		max_y = np.max(enlarged_bbox[:, 1])
		plot_bbox(prev_bbox, 'blue')
		plot_bbox(enlarged_bbox, 'blue', '--')

		keypoints = self.map.get_keypoints(Polygon(enlarged_bbox))
		keypoint_dict = dict()
		for kp in keypoints:
			index = kp.index
			pt = kp.point
			if index not in keypoint_dict.keys():
				keypoint_dict[index] = []
			keypoint_dict[index].append(pt)
		
		colors = ['red', 'blue', 'green', 'yellow', 'purple', 'black']
		for key, value in keypoint_dict.items():
			x = [v[0] for v in value]
			y = [v[1] for v in value]
			plt.scatter(x, y, color=colors[key % len(colors)], alpha=0.1)

		plt.ylim(max_y+50, min_y-50)
		plt.show()

	@staticmethod
	def get_color(score):
		color = 'grey'
		if score > 0.3:
			color = 'blue'
		if score > 0.5:
			color = 'red'
		return color

	def plot_map(self):
		
		kp_dict = dict()
		
		for kp in self.map.keypoints:
			color = self.get_color(kp.score)
			if color not in kp_dict.keys():
				kp_dict[color] = []
			kp_dict[color].append(kp.point)
		
		for color, points in kp_dict.items():
			x = [p[0] for p in points]
			y = [p[1] for p in points]
			plt.scatter(x, y, color=color, alpha=0.1)

		min_y = float('inf')
		max_y = float('-inf')
		for i in range(len(self.map_bboxes)):
			bbox = self.map_bboxes[i]
			
			plot_bbox(bbox, 'blue')
			
			min_y_bbox = np.min(bbox[:, 1])
			max_y_bbox = np.max(bbox[:, 1])
			min_y = min(min_y, min_y_bbox)
			max_y = max(max_y, max_y_bbox)
		plt.ylim(max_y+50, min_y-50)
		plt.show()

	@staticmethod
	def find_matches(prev_descriptors, new_descriptors):
		matches = flann.knnMatch(prev_descriptors, new_descriptors, k=2)
		good_matches = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good_matches.append(m)
		return good_matches

	@staticmethod
	def find_homography(src_pts, dst_pts):
		src_pts = np.float32(src_pts).reshape(-1, 1, 2)
		dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

		H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

		return H

def plot_bbox(bbox, color='red', style='-'):
	for i in range(len(bbox)):
		current_point = bbox[i]
		next_point = bbox[(i + 1) % len(bbox)]
		plt.plot([current_point[0], next_point[0]], [current_point[1], next_point[1]], style, color=color)

def get_jpg_files(directory):
	pattern = os.path.join(directory, '*.[jJ][pP][gG]')
	jpg_files = glob.glob(pattern)
	return jpg_files

if __name__ == "__main__":
	jpg_files = get_jpg_files('./data/images2')
	queue = list(sorted(jpg_files))

	g = Generator(queue)
	g.run()