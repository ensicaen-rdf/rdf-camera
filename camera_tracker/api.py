from itertools import tee
import cv2

from flask import Blueprint, Response, request
from .OCV import *

bp = Blueprint('api', __name__, url_prefix='/v1')   

@bp.route('/', methods=('GET',))
def api():
    return 'camera_tracker api'

@bp.route('/addPerson', methods=('POST',))
def addPerson():    
    pass

@bp.route('/video_feed', methods=("GET",))
def video_feed():
    idNational = request.args.get('id')
    print(idNational, idNational in users)
    if idNational in users:
        try:
            return Response(listSources[users[idNational]].detectFaces(), mimetype='multipart/x-mixed-replace; boundary=frame')
        except StopIteration:
            return "User not found on camera"

    return "User not found or id not provided"
