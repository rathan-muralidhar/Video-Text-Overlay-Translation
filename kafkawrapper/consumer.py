import logging
from kafka import KafkaConsumer, TopicPartition
from config.config import kafka_bootstrap_server_host,video_convert_consumer_grp,video_convert_input_topic,video_convert_output_topic,total_no_of_partitions
import json
import random
import string

log = logging.getLogger('file')

# Method to instantiate the kafka consumer
def instantiate(topics):
    consumer = KafkaConsumer(*topics,
                             bootstrap_servers=list(str(kafka_bootstrap_server_host).split(",")),
                             api_version=(1, 0, 0),
                             group_id=video_convert_consumer_grp,
                             auto_offset_reset='latest',
                             enable_auto_commit=True,
                             value_deserializer=lambda x: handle_json(x))
    return consumer


# For all the topics, returns a list of TopicPartition Objects
def get_topic_paritions(topics):
    topic_paritions = []
    for topic in topics:
        for partition in range(0, total_no_of_partitions):
            tp = TopicPartition(topic, partition)
            topic_paritions.append(tp)
    return topic_paritions

# Method to read and process the requests from the kafka queue
def consume():
    try:
        topics = [video_convert_input_topic]
        consumer = instantiate(topics)

        rand_str = ''.join(random.choice(string.ascii_letters) for i in range(4))
        prefix = "Video-Convert-Core-" + "(" + rand_str + ")"
        #log.info(prefix + " Running..........")
        while True:
            for msg in consumer:
                data = {}
                try:
                    data = msg.value
                    if data:
                        if msg.topic == video_convert_input_topic:
                            #log.info(prefix + " | Received on Topic: " + msg.topic + " | Partition: " + str(msg.partition), data)
                            pass
                    else:
                        break
                except Exception as e:
                    log.exception(prefix + " Exception in translator while consuming: " + str(e))
    except Exception as e:
        log.exception("Exception while starting the translator consumer: " + str(e))

# Method that provides a deserialiser for the kafka record.
def handle_json(x):
    try:
        return json.loads(x.decode('utf-8'))
    except Exception as e:
        log.exception("Exception while deserialising: "+e)
        return {}