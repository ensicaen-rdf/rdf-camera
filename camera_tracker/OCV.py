import os
import cv2
import numpy as np
from PIL import Image

users = {
    "1a2b3c4d": -1
}

def getIds():
    names = []
    with open('./camera_tracker/Id.txt', "r+") as f:
        for line in f:
            data = line.strip().split(";")
            names.append({"name": data[0], "id": data[1]})

    return names


def getTrainerFile():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r'./camera_tracker/trainer/trainer.yml')
    return recognizer

class Source:
    def __init__(self, url) -> None:
        self.url = url
        self.launched = False
        self.lastWho = "None"

    def detectFaces(self):
        names = getIds()
        recognizer = getTrainerFile()
        faceCascade = cv2.CascadeClassifier(r'./camera_tracker/haarcascade_frontalface_default.xml')

        id = 0
        cam = cv2.VideoCapture(self.url)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            success, img = cam.read()
            if not success:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            who = "no-one"
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    who = names[id]["name"]
                    users[names[id]["id"]] = self.url
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    who = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(who), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            self.lastWho = who
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cam.release()

def addFace(id):

    def getImagesAndLabels(path):
        imagePaths  = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids         = []

        for imagePath in imagePaths:
            img_numpy = np.array(Image.open(imagePath).convert('L'), 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    #close each sources
    for camera in listSources:
        camera.release()
    
    #train the model
    faces, ids = getImagesAndLabels("./camera_tracker/dataset")
    detector = cv2.CascadeClassifier(r'./camera_tracker/haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write(r'./camera_tracker/trainer/trainer.yml')


listSources = [Source(1), Source(0)]
