from flask import Flask, jsonify, request
from config.config import INPUT_DIR, ocr_mode
import os
from utilities.cv2utils import CV2_HELPER
from services.ocrhandler import OCR_HANDLER
import logging
app = Flask(__name__)

# Video Translate Workflow

log = logging.getLogger('file')

@app.route('/v1/videotranslate', methods=["POST"])
def video_translate_workflow():
    data = request.get_json()
    log.info(data)
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
