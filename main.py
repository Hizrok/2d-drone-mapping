#!/usr/bin/python3

import cv2 as cv
import numpy as np
import math

DIR = "./data/images/DJI_{}.jpg"

TOTAL = 20
OFFSET = 1000

def run():
	image_a = cv.imread(get_image_path(1))
	
	height, width = image_a.shape[:2]
	offset_x = (width+OFFSET) / 2 - width / 2
	offset_y = (height+OFFSET) / 2 - height / 2
	translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
	res = (width+OFFSET, height+OFFSET)
	rect = get_rect(image_a, translation_matrix)
	image_a = cv.warpPerspective(image_a, translation_matrix, res)

	for i in range(2, TOTAL+1):	
		print(f"[{i}] detecting features from the main image")
		key_points_a, descriptor_a = detect_features(image_a)
		image_b = cv.imread(get_image_path(i))
		
		print(f"[{i}] detecting features from the new image")
		key_points_b, descriptor_b = detect_features(image_b)
		
		print(f"[{i}] matching features")
		matches = match_features(descriptor_a, descriptor_b)
		
		print(f"[{i}] calculating homography")
		homography = get_homography(key_points_a, key_points_b, matches)

		print(f"[{i}] warping image")
		warped_image = cv.warpPerspective(image_b, homography, (image_a.shape[1], image_a.shape[0]))

		overlayed_image = np.where(warped_image != 0, warped_image, image_a)
		image_a = overlayed_image

		print(f"rect: {rect}")
		new_rect = get_rect(image_b, homography)
		print(f"new rect: {new_rect}")
		
		rect = merge_rects(rect, new_rect)
		print(f"merged: {rect}")
		
		print(f"old res: {res}")
		res = get_new_res(rect)
		print(f"new_res: {res}")
		
		image_a, rect = center(image_a, rect, res)

		cv.imwrite(f"out.jpg", image_a)
		# input("Press Enter to continue...")

def center(image, rect, res):
	min_x, min_y, max_x, max_y = rect
	w, h = (max_x-min_x, max_y-min_y)
	W, H = res

	offset_x = round(W / 2 - w / 2 - min_x)
	offset_y = round(H / 2 - h / 2 - min_y)
	translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
	warped_image = cv.warpAffine(image, translation_matrix, res)
	new_rect = (min_x+offset_x, min_y+offset_y, max_x+offset_x, max_y+offset_y)
	return warped_image, new_rect

def get_new_res(rect):
	min_x, min_y, max_x, max_y = rect
	width = max_x-min_x
	height = max_y-min_y
	return (width+OFFSET, height+OFFSET)

def get_rect(image, matrix):
	height, width = image.shape[:2]
	corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
	warped_corners = cv.perspectiveTransform(corners, matrix)
	min_x = round(np.min(warped_corners[:, :, 0]))
	min_y = round(np.min(warped_corners[:, :, 1]))
	max_x = round(np.max(warped_corners[:, :, 0])+1)
	max_y = round(np.max(warped_corners[:, :, 1])+1)
	return (min_x, min_y, max_x, max_y)

def merge_rects(rect_a, rect_b):
	a_min_x, a_min_y, a_max_x, a_max_y = rect_a
	b_min_x, b_min_y, b_max_x, b_max_y = rect_b
	min_x = min(a_min_x, b_min_x)
	min_y = min(a_min_y, b_min_y)
	max_x = max(a_max_x, b_max_x)
	max_y = max(a_max_y, b_max_y)
	return (min_x, min_y, max_x, max_y)

def get_homography(key_points_a, key_points_b, matches):
	# Extract keypoints from the good matches
	src_pts = np.float32([key_points_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([key_points_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

	# Find homography
	homography, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC)

	return homography

def match_features(descriptor_a, descriptor_b):
	flann = cv.FlannBasedMatcher()
	matches = flann.knnMatch(descriptor_a, descriptor_b, k=2)
	good_matches = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good_matches.append(m)
	# print(len(good_matches))
	return good_matches

def detect_features(image):
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	sift = cv.SIFT_create()
	key_points, descriptor = sift.detectAndCompute(gray, None)
	return (key_points, descriptor)

def resize_image(image, factor=2):
	height, width = image.shape[:2]
	new_height = int(height / factor)
	new_width = int(width / factor)
	resized_image = cv.resize(image, (new_width, new_height))
	return resized_image

def show_image(image, description="image"):
	cv.imshow(description, image)
	cv.waitKey(0)
	cv.destroyAllWindows()

def get_image_path(index):
	str_index = str(index)
	while len(str_index) < 4:
		str_index = f"0{str_index}"
	return DIR.format(str_index)

if __name__ == "__main__":
	run()