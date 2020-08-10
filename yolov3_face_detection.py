import argparse
import sys
import os
import numpy as np
from PIL import Image
import cv2
from utils import *

class Face_detection:

	def __init__(self):
		print("[INFO] Initialized Face Detection")

		self.net = cv2.dnn.readNetFromDarknet('path to yolov3 cfg file', 'yolov3 weights files')
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		# for gpu usage change cv2.dnn.DNN_TARGET_CPU to cv2.dnn.DNN_TARGET_OPENCL
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
		
		

	def predict(self, img, min_score=0.4):
		"""
		Input : img (str, io.BytesIO, np.ndarray): image path or image
		        min_score (float)                : confidence threshold
		---------------------------------------------------------------
		Output: output (list of tuples)          : (bbox, score, class_name)
		        img (np.ndarray)                 : RGB image
        """

		if type(img) is not np.ndarray:
			img = Image.open(img)
			# img = np.array(img)
		
		blob = cv2.dnn.blobFromImage(img, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
		                     [0, 0, 0], 1, crop=False)
		self.net.setInput(blob)

		outs = self.net.forward(get_outputs_names(self.net))

		# Remove the bounding boxes with low confidence
		faces,confidence = post_process(img, outs, CONF_THRESHOLD, NMS_THRESHOLD)
		output = []
		for i in range(len(faces)):
			bbox = faces[i]
			score = confidence[i]
			if score >= min_score:
				output.append((bbox,score,'Face'))
		return output,img

