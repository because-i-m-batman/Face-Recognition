# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
# from mtcnn.mtcnn import MTCNN
from FaceDetection import Face_detection
import cv2

# extract a single face from a given photograph

def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()

	results = detector.detect_faces(pixels)
	
	if not results:
		Faces_Array = pixels

	else:
		x1, y1, width, height = results[0]['box']


		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]

		# resize pixels to the model size
		
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		Faces_Array = face_array
	return Faces_Array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# img = cv2.imread(path)
		# face = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		# face = cv2.resize(face,(160,160))
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]

		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('Path to Image directory')
print(trainX.shape, trainy.shape)
# load test dataset

# save arrays to one file in compressed format
savez_compressed('faces.npz', trainX, trainy)