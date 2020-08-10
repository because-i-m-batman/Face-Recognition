import cv2
from keras.models import load_model
from numpy import expand_dims
from numpy import asarray
from yolov3_face_detection import Face_detection
from tqdm import tqdm
from keras.preprocessing import image
import operator
import itertools

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    samples = expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)
    return yhat[0]

c = Face_detection()

model_classification = load_model('path to trained face model using face_training.py file')
model_embedding = load_model('path to facenet keras model')

#Video Reading
cap = cv2.VideoCapture('path to input video file')

output_path = 'out_video1.mp4'
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="[INFO] Processing video")

# Create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
count_frames = 0
while cap.isOpened():
        # Get next frame in video

	ret, orig_image = cap.read()
	if not ret: # Reached end of video
		break
	orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)



		# get faces using yolov3 face detection
	output,img = c.predict(orig_image,0.7)


	for out in output:
		x1, y1, width, height = out[0]

		try:
			image = orig_image[y1:height+5,x1:width+2]


			image = cv2.resize(image,(160,160))
			# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

			#create embeddings of faces obtain
			embedding = get_embedding(model_embedding,image)
			embedding = expand_dims(embedding, axis=0)
			out = model_classification.predict(embedding)
			out = out.tolist()
			flatten = itertools.chain.from_iterable
			out = list(flatten(out))
			person_class, confidence = max(enumerate(out), key=operator.itemgetter(1))


			if person_class == 0 and confidence> 0.7:
				person = 'Chandler'
			elif person_class == 1 and confidence> 0.7:
				person = 'Joey'
			elif person_class == 2 and confidence> 0.7:
				person = 'Monica'
			elif person_class == 3 and confidence> 0.7:
				person = 'Phoebe'
			elif person_class == 4 and confidence> 0.7:
				person = 'Rachel'
			elif person_class == 5 and confidence> 0.7:
				person = 'Ross'
			else:
				person = "Unknown" 


			cv2.rectangle(orig_image, (x1, y1), (width+2,height+5), (255, 255, 0), 2)
			cv2.putText(orig_image,person,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 255), 1)







		except:
			print('No face detected')
	orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
	count_frames += 1
	pbar.update(1)
	cv2.imshow("Frame", orig_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	video.write(orig_image)
	while count_frames < total_frames:
		count_frames += 1
		pbar.update(1)
pbar.close()
cap.release()
video.release()
cv2.destroyAllWindows()
