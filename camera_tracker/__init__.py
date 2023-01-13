from threading import Thread, Event, current_thread
from queue import Queue
import cv2
import os
import ffmpegcv
import requests as req

from flask import Flask, request, Response, url_for, redirect, g

users = {}
fluxs = {}
stopEvent = None

def getIds():
    names = []
    with open('./camera_tracker/Id.txt', "r+") as f:
        for line in f:
            data = line.strip().split(";")
            names.append({"name": data[0], "id": data[1]})

    return names


def detectFaces(urlQueue: Queue, users: Queue, flux: Queue, stopEvent: Event):
    names = getIds()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r'./camera_tracker/trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(
        r'./camera_tracker/haarcascade_frontalface_default.xml')

    listIdOnCamera = []
    index, url = urlQueue.get()
    print(f"Thread started {current_thread().ident} on {url}")
    cam = ffmpegcv.VideoCaptureCAM(camname=url,
                                   camsize=(640, 480),
                                   pix_fmt='bgr24',
                                   crop_xywh=None,
                                   resize=None,
                                   resize_keepratio=True,
                                   resize_keepratioalign='center')
    # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not cam.isopened:
        print("Error opening camera")
        return
    print("Camera opened")
    # cam.set(3, 640)  # set video widht
    # cam.set(4, 480)  # set video height

    minW = 0.1 * 640 #cam.get(3)
    minH = 0.1 * 480 #cam.get(4)

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
                nationalId = names[id]["id"]
                stillOnCamera.append(id)
                if id not in listIdOnCamera:
                    u = users.get()
                    u[nationalId] = index
                    print(f"User {nationalId} detected on camera {index}")
                    users.put(u)

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
                u = users.get()
                u[nationalId] = -1
                users.put(u)
                listIdOnCamera.remove(id)

        ret, frame = cv2.imencode('.jpg', img)
        flux.put(frame.tobytes())

    cam.release()

def launch(listUrl):

    fluxs = {}
    users = {}

    for name in getIds():
        users[name["id"]] = -1

    for index, url in listUrl:
        fluxs[index] = Queue()

    urlQueue = Queue()
    for index, url in listUrl:
        urlQueue.put((index, url))

    stopEvent = Event()
    usersQueue = Queue()
    usersQueue.put(users)

    for index, url in listUrl:
        Thread(target=detectFaces, args=(
            urlQueue, usersQueue, fluxs[index], stopEvent)).start()

    return fluxs, stopEvent, users

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

    def gen_frames(q: Queue):
        while True:
            frame = q.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/camera_feed', methods=("GET",))
    def camera_feed():
        global fluxs
        id = int(request.args.get('id'))
        if id < len(fluxs):
            return Response(gen_frames(fluxs[id]), mimetype='multipart/x-mixed-replace; boundary=frame')
        return "Camera not found"

    @app.route('/person_feed', methods=("GET",))
    def video_feed():
        global users
        idNational = request.args.get('id')

        if idNational in users.keys():
            print(users)
            if users[idNational] != -1:
                print(f"redirect to camera {users[idNational]}")
                return redirect(url_for('camera_feed', id=users[idNational]))

            return "User not found on camera"

        return "User not found or id not provided"

    fl, st, us = launch(
        [(0, "http://192.168.3.145:4747/video"),
         (1, "http://192.168.3.34:4747/video")])
    global users, fluxs
    users = us
    fluxs = fl

    return app
