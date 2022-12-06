import logging
from config.config import app_host,app_port
from controller.controller import app
from multiprocessing import Process
from kafkawrapper.consumer import consume

log = logging.getLogger('file')

# Starts the kafka consumer in a different thread
def start_consumer():
    with app.test_request_context():
        try:
            consumer_process = Process(target=consume)
            consumer_process.start()
        except Exception as e:
            log.exception("Exception while starting the Translator kafka consumers: " + str(e))

if __name__ == '__main__':
    start_consumer()
    app.run(host=app_host, port=app_port, threaded=True)
