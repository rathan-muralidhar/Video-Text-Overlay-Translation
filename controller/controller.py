from flask import Flask, jsonify, request

app = Flask(__name__)

#Video Translate Workflow
@app.route('/v1/videotranslate', methods=["POST"])
def video_translate_workflow():
    response = {}
    return jsonify(response), 200


# Health endpoint
@app.route('/health', methods=["GET"])
def health():
    response = {"code": "200", "status": "ACTIVE"}
    return jsonify(response)