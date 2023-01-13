# Serveur de Surveillance

## Requirement

* All dependencies are in the requirements.txt file. To install juste use ```pip3 install -r requirements.txt```. We recommand using a virtual env.

## Run the project

### Train the model to detect faces :

1. Before launching the server, you need to add faces that you want the model to recognize.

To do so, use ```python3 camera_tracker/01_face_dataset.py```. This part require a webcam connected to the computer. If no webcam is available, you can use a webcam connected to internet (see DroidCam to use your phone as a webcam). You will need to adapt the line ```cam = cv2.VideoCapture(0)``` in the ```detectFaces``` function like in the ```camera_tracker/__init__.py``` file.

2. Then, use ```python3 camera_tracker/02_face_training.py``` to train the model with the face previously taken.

3. Lastly, for each faces that you trained for, you'll need to add a line in the ```Id.txt``` file, corresponding to the name of the faces inputted in the model, associated to an id. The name need to be in the same order as when inputted. The id will be used to access the video flux of the correspondig camera if someone is detected.

**You are now ready to launch the flask server!**

### Launch the Flask server

To run the flask server locally, you need to export one variables as follow:
```bash
export FLASK_APP=camera_tracker
```

Then the server should be accessible at http://127.0.0.1:5000/

To access the video flux of a webcam, go to ```/video_feed/<id>``` where id is the index of the camera source.

You can add camera to the server by modifying the ```camera_tracker/__init__.py``` file at the end of the file.

You can then prompt the server the video flux of any id, where the id is the id used in the text file previously written. If this person is detected on one camera, it will redirect you on the video flux of the good camera.