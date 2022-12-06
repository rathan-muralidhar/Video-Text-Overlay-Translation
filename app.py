import logging
from logging.config import dictConfig
from config.config import app_host,app_port
from controller.controller import app
from multiprocessing import Process
from kafkawrapper.consumer import consume

log = logging.getLogger('file')

# Starts the kafka consumer in a different thread
def start_consumer():
    pass
    # with app.test_request_context():
    #     try:
    #         consumer_process = Process(target=consume)
    #         consumer_process.start()
    #     except Exception as e:
    #         log.exception("Exception while starting the Translator kafka consumers: " + str(e))

if __name__ == '__main__':
    start_consumer()
    app.run(host=app_host, port=app_port, threaded=True, debug=False)

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