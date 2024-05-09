#!/usr/bin/python3

##
# @author Jan Kapsa
# @date 9.5.2024
# @file strategy.py contains all the calculations

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
import math

from lib.extractor import Extractor
from lib.matcher import Matcher

## @class Keypoint
# @brief represents a point in space used for matching againts other images
class Keypoint:
	
	## @brief initialize the keypoint class
	# @param index index of image keypoint belongs to
	# @param point (x, y) tuple representing the keypoint in space
	# @param descriptor descriptor of a feature this keypoint was made from (used for matching)
	def __init__(self, index, point, descriptor):
		self.index = index
		self.point = point
		self.descriptor = descriptor
		self.value = 0.5

	## @brief increses or decreses the value of a keypoint depending on if the keypoint was used for matching or not
	# @param is_match bool value representing if the keypoint was used for matching or not
	def score(self, is_match=False):
		if is_match:
			self.value *= 1.5
		else:
			self.value *= 0.7

## @class Map
# @brief a list of keypoints and operations on that list
class Map:
	
	## @brief initializes keypoint map
	def __init__(self):
		self.keypoints: list[Keypoint] = []

	## @brief adds new keypoints to map
	def add_keypoints(self, new_keypoints: list[Keypoint]):
		self.keypoints += new_keypoints

	## @brief gets keypoints inside supplied polygon
	# @param polygon polygon window to get the keypoints from
	# @return list of keypoints inside polygon
	def get_keypoints(self, polygon: Polygon):
		keypoints: list[Keypoint] = []

		for keypoint in self.keypoints:
			if keypoint.value > 0.3:
				point = Point(keypoint.point)
				if polygon.contains(point):
					keypoints.append(keypoint)

		return keypoints

## @class Strategy
# @brief parent class containing common variables
class Strategy:
	
	## @brief initializes common values
	def __init__(self):
		
		self.mosaic = None
		self.offset_matrix = None
		self.width = None
		self.height = None
		self.bounding_boxes = []
		self.matches = []

		self.extractor = Extractor(4, 1000)
		self.matcher = Matcher()		

	def initialize(self, image_path):
		pass
	
	def new_image(self, image_path, index):
		pass
	
	## @brief handles the offset of the new image
	# @param homography calculated homography matrix
	# @param bbox default image bounding box
	# @param img image with RGB values
	def offset(self, homography, bbox, img):
		image_homography = np.matmul(self.offset_matrix, homography)
		
		transformed_bbox = cv2.perspectiveTransform(bbox.reshape(-1, 1, 2), image_homography)
		transformed_bbox = np.squeeze(transformed_bbox)

		min_x = np.min(transformed_bbox[:, 0])
		max_x = np.max(transformed_bbox[:, 0])
		min_y = np.min(transformed_bbox[:, 1])
		max_y = np.max(transformed_bbox[:, 1])

		min_x = math.floor(min_x) if min_x < 0 else math.ceil(min_x)
		max_x = math.floor(max_x) if max_x < 0 else math.ceil(max_x)
		min_y = math.floor(min_y) if min_y < 0 else math.ceil(min_y)
		max_y = math.floor(max_y) if max_y < 0 else math.ceil(max_y)

		offset_x = 0
		offset_y = 0

		if min_x < 0:
			offset_x = abs(min_x)
			max_x += offset_x
			self.width += offset_x
			min_x = 0

		if min_y < 0:
			offset_y = abs(min_y)
			max_y += offset_y
			self.height += offset_y
			min_y = 0

		res_changed = False
		offset_found = False

		if self.width < max_x:
			self.width = max_x
			res_changed = True
		if self.height < max_y:
			self.height = max_y
			res_changed = True

		offset_matrix = np.float32([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]
		])

		if offset_x != 0 or offset_y != 0:
			offset_found = True
			offset_matrix = np.float32([
				[1, 0, offset_x],
				[0, 1, offset_y],
				[0, 0, 1]
			])
			
			self.mosaic = cv2.warpPerspective(self.mosaic, offset_matrix, (self.width, self.height))
			image_homography = np.matmul(offset_matrix, image_homography)
			self.offset_matrix = np.matmul(offset_matrix, self.offset_matrix)

		if res_changed and not offset_found:
			self.mosaic = cv2.resize(self.mosaic, (self.width, self.height))

		img = cv2.warpPerspective(img, image_homography, (self.width, self.height))
		_, binary_mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

		transformed_bbox = cv2.perspectiveTransform(bbox.reshape(-1, 1, 2), image_homography)
		transformed_bbox = np.squeeze(transformed_bbox)

		min_x = np.min(transformed_bbox[:, 0])
		max_x = np.max(transformed_bbox[:, 0])
		min_y = np.min(transformed_bbox[:, 1])
		max_y = np.max(transformed_bbox[:, 1])

		min_x = math.floor(min_x) if min_x < 0 else math.ceil(min_x)
		max_x = math.floor(max_x) if max_x < 0 else math.ceil(max_x)
		min_y = math.floor(min_y) if min_y < 0 else math.ceil(min_y)
		max_y = math.floor(max_y) if max_y < 0 else math.ceil(max_y)

		for y in range(min_y, max_y):
			for x in range(min_x, max_x):
				if binary_mask[y, x] == 255:
					self.mosaic[y, x] = img[y, x]

	@staticmethod
	def find_homography(src_pts, dst_pts):
		src_pts = np.float32(src_pts).reshape(-1, 1, 2)
		dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

		H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

		return H

## @class KeypointMapStrategy
# @brief implements keypoint map strategy, inherits from Strategy
class KeypointMapStrategy(Strategy):

	## @brief initialize keypoint map
	def __init__(self):
		super().__init__()

		self.map = Map()
		self.prev_bounding_box = None

	## @brief initializes keypoint map strategy
	def initialize(self, image_path):
		print(f"image: {image_path}")
		
		img, features, descriptors = self.extractor.extract(image_path)

		h, w = img.shape[:2]
		bounding_box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
		self.prev_bounding_box = bounding_box
		self.bounding_boxes.append(bounding_box)

		self.mosaic = img
		self.width = w
		self.height = h
		self.offset_matrix = [[1,0,0], [0,1,0], [0,0,1]]

		keypoints = [Keypoint(0, features[i].pt, descriptors[i]) for i in range(len(features))]
		self.map.add_keypoints(keypoints)

	## @brief analyzing and warping new incoming images while using the keypoint map
	def new_image(self, image_path, index):
		print(f"image: {image_path}")

		img, features, descriptors = self.extractor.extract(image_path)

		h, w = img.shape[:2]
		bounding_box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

		# calculating the prediction window
		scale_factor = 1.8
		prediction_window = []
		prev_bbox_polygon = Polygon(self.prev_bounding_box)
		centroid = np.array([prev_bbox_polygon.centroid.x, prev_bbox_polygon.centroid.y])
		for point in self.prev_bounding_box:
			vector = np.array([point[0] - centroid[0], point[1] - centroid[1]])
			norm = np.linalg.norm(vector)
			norm_vector = vector / norm
			dist = np.linalg.norm(point - centroid)
			prediction_window.append(centroid + (norm_vector * dist * scale_factor))
		prediction_window = np.array(prediction_window, dtype=np.float32)
		
		keypoints: list[Keypoint] = self.map.get_keypoints(Polygon(prediction_window))
		map_points = [keypoints[i].point for i in range(len(keypoints))]
		map_descriptors = np.float32([keypoints[i].descriptor for i in range(len(keypoints))])

		# find matches
		matches = self.matcher.find_matches(map_descriptors, descriptors)
		src_pts = [features[m.trainIdx].pt for m in matches]
		dst_pts = [map_points[m.queryIdx] for m in matches]

		print(f"matches: {len(matches)}")
		self.matches.append(len(matches))

		# find homography
		homography = self.find_homography(src_pts, dst_pts)

		transformed_bbox = cv2.perspectiveTransform(bounding_box.reshape(-1, 1, 2), homography)
		transformed_bbox = np.squeeze(transformed_bbox)

		self.prev_bounding_box = transformed_bbox
		self.bounding_boxes.append(transformed_bbox)

		# score points in the intersection
		indexes: list[int] = list(sorted(list(set([m.queryIdx for m in matches])), reverse=True))
		for i in indexes:
			keypoints[i].score(True)
			keypoints.pop(i)
			map_points.pop(i)	
		poly1 = Polygon(prediction_window)
		poly2 = Polygon(transformed_bbox)
		try:
			intersection: Polygon = poly1.intersection(poly2)
		except:
			exit("[error] could not find polygon intersection, exiting...")
		for i in range(len(map_points)):
			point = Point(map_points[i])
			if intersection.contains(point):
				keypoints[i].score()

		# add new keypoints to map
		indexes = list(sorted(list(set([m.trainIdx for m in matches])), reverse=True))
		features = list(features)
		descriptors = list(descriptors)
		for i in indexes:
			features.pop(i)
			descriptors.pop(i)
		new_points = np.array([kp.pt for kp in features])
		new_points = cv2.perspectiveTransform(new_points.reshape(-1, 1, 2), homography)
		new_points = np.squeeze(new_points)
		self.map.add_keypoints([Keypoint(index, new_points[i], descriptors[i]) for i in range(len(descriptors))])

		self.offset(homography, bounding_box, img)

## @class PrevImageStrategy
# @brief implements previous image strategy, inherits from Strategy
class PrevImageStrategy(Strategy):
	
	def __init__(self):
		super().__init__()

		self.prev_features = None
		self.prev_descriptors = None
		self.prev_homography = None

	## @brief initializes the strategy, sets first values
	def initialize(self, image_path):
		print(f"image: {image_path}")

		img, features, descriptors = self.extractor.extract(image_path)

		h, w = img.shape[:2]
		bounding_box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
		self.prev_bounding_box = bounding_box
		self.bounding_boxes.append(bounding_box)

		self.mosaic = img
		self.width = w
		self.height = h
		self.offset_matrix = [[1,0,0], [0,1,0], [0,0,1]]
		
		self.prev_features = features
		self.prev_descriptors = descriptors
		self.prev_homography = [[1,0,0], [0,1,0], [0,0,1]]

		pass

	## @brief analyzing and warping new incoming images based on their mathces with the previous image
	def new_image(self, image_path, index):
		print(f"image: {image_path}")
		
		img, features, descriptors = self.extractor.extract(image_path)

		h, w = img.shape[:2]
		bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

		matches = self.matcher.find_matches(self.prev_descriptors, descriptors)
		print("matches:", len(matches))
		self.matches.append(len(matches))

		src_pts = [features[m.trainIdx].pt for m in matches]
		dst_pts = [self.prev_features[m.queryIdx].pt for m in matches]

		homography = self.find_homography(src_pts, dst_pts)
		homography = np.matmul(self.prev_homography, homography)

		transformed_bbox = cv2.perspectiveTransform(bbox.reshape(-1, 1, 2), homography)
		transformed_bbox = np.squeeze(transformed_bbox)
		self.bounding_boxes.append(transformed_bbox)

		self.offset(homography, bbox, img)

		self.prev_features = features
		self.prev_descriptors = descriptors
		self.prev_homography = homography