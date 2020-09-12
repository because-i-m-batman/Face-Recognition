# Face-Recognition

# Usage

1. Collect database

    -> Arrange the Images in given way:

        1.Person1 (Folder Contains Images of Person1)

        2.Person2 (Folder Contains Images of Person2)

        3.Person3 (Folder Contains Images of Person3)

2. Run face_detection.py,it will create faces.npz file

3. Run face_embedding.py,it will create embeddings.npz file which contains face embedding (128 vector for each face)

    -> To run this file download facenet keras model(facenet_keras.h5) from here : https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

4. Run face_training.py,it will create trained face embedded model(model.hdf5)

5. Run face_recognition.py,it will create a video of recognized faces

    -> To run this file,download yolov3 weights and cfg file ,this face_recognition.py file use yolov3_face_detection.py file for face_detection because its faster.

    -> download weights file from here :https://drive.google.com/file/d/1zU_n5CwnGfYgFNLQ1JZlsl-rHjPV-kmp/view

    -> download cfg file from here : https://github.com/sthanhng/yoloface/tree/master/cfg
    
6. Output Video file: https://drive.google.com/file/d/1cf9nBBoXfQByGlWYj1JkZF1IShigENw9/view?usp=sharing
