from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
data = load('faces.npz',allow_pickle=True)
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
count = 0
for face_pixels in trainX:
	count+=1
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
	print('Face {}'.format(count))

newTrainX = asarray(newTrainX)
# print(newTrainX.shape)
# convert each face in the test set to an embedding

# save arrays to one file in compressed format
savez_compressed('embeddings.npz', newTrainX, trainy)