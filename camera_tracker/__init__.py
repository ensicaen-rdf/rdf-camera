from threading import Thread, Event, current_thread
from queue import Queue
import cv2
import os
import atexit

from flask import Flask, request, Response

users = {
    "1a2b3c4d": -1,
    "5e6f7g8h": -1,
    "9i10j11k12l": -1,
}
fluxs = []
threads = []
stopEvent = Event


def getIds():
    names = []
    with open('./camera_tracker/Id.txt', "r+") as f:
        for line in f:
            data = line.strip().split(";")
            names.append({"name": data[0], "id": data[1]})

    return names


def detectFaces(urlQueue: Queue, eventQueue: Queue, flux: Queue, stopEvent: Event):
    names = getIds()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r'./camera_tracker/trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(
        r'./camera_tracker/haarcascade_frontalface_default.xml')

    listIdOnCamera = []
    url = urlQueue.get()
    print(current_thread().ident, url)
    cam = cv2.VideoCapture(url)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while stopEvent.is_set:
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

        stillOnCamera = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                who = names[id]["name"]
                stillOnCamera.append(id)
                if id not in listIdOnCamera:
                    eventQueue.put({"name": who, "id": id, "source": url, "confidence": confidence})
                    listIdOnCamera.append(id)
            else:
                who = "unknown"

            cv2.putText(img, str(who), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(f"  {round(100 - confidence)}%"), (x + 5, y + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        for id in listIdOnCamera:
            if id in stillOnCamera:
                pass
            else:
                eventQueue.put({"name": who, "id": id, "source": url, "confidence": None})
                listIdOnCamera.remove(id)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        flux.put(b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()


def handleEvent(eventQueue: Queue, stopEvent: Event):

    while not stopEvent.is_set():
        event = eventQueue.get()
        if event["confidence"] is None:
            users[event["id"]] = -1
        else:
            users[event["id"]] = event["source"]
        print(event)


def launch(listUrl):

    global fluxs, eventQueue, stopEvent, threads
    for i in range(len(listUrl)):
        fluxs.append(Queue())

    urlQueue = Queue()
    for url in listUrl:
        urlQueue.put(url)

    eventQueue = Queue()
    stopEvent = Event()

    for i in range(len(listUrl)):
        threads.append(Thread(target=detectFaces, args=(
            urlQueue, eventQueue, fluxs[i], stopEvent)))

    for thread in threads:
        print("Starting thread", thread.ident)
        thread.start()

    return fluxs, eventQueue, stopEvent, threads

def stopServer():
    global stopEvent, threads
    stopEvent.set()
    for thread in threads:
        thread.join()

def create_app(config = None):

    global users, fluxs, stopEvent, threads
    app = Flask(__name__, instance_relative_config = True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
    )

    if config is None:
        app.config.from_pyfile('config.py', silent = True)
    else:
        app.config.from_mapping(config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/', methods=('GET',))
    def api():
        return 'camera_tracker api'

    @app.route('/video_feed', methods=("GET",))
    def video_feed():
        idNational = request.args.get('id')

        if idNational in users:
            if users[idNational] != -1:
                return Response(fluxs[users[idNational]], mimetype='multipart/x-mixed-replace; boundary=frame')

            return "User not found on camera"

        return "User not found or id not provided"

    fluxs, eventQueue, stopEvent, threads = launch([0])
    Thread(target=handleEvent, args=(eventQueue, stopEvent)).start()

    atexit.register(stopServer)
    return app
