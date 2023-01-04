from flask import Flask, jsonify, request
from config.config import INPUT_DIR, ocr_mode
import os
from utilities.cv2utils import CV2_HELPER
from services.ocrhandler import OCR_HANDLER
import logging
from logging.config import dictConfig

app = Flask(__name__)

# Video Translate Workflow

log = logging.getLogger('file')

@app.route('/v1/videotranslate', methods=["POST"])
def video_translate_workflow():
    data = request.get_json()
    log.info(f"Request {data}")
    filename = data['filename']
    if os.path.isfile(filename):
        ocr_handler = OCR_HANDLER(filename, CV2_HELPER(), ocr_mode)
        ocr_handler.process_frames()
        ocr_handler.assemble_video()
    response = {}
    return jsonify(response), 200


# Health endpoint
@app.route('/health', methods=["GET"])
def health():
    response = {"code": "200", "status": "ACTIVE"}
    return jsonify(response)

# Log config
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] {%(filename)s:%(lineno)d} %(threadName)s %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {
        'info': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'info.log'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        'file': {
            'level': 'DEBUG',
            'handlers': ['info', 'console'],
            'propagate': ''
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['info', 'console']
    }
})