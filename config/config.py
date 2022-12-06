import os

#kafka-configs
kafka_bootstrap_server_host = os.environ.get('KAFKA_BOOTSTRAP_SERVER_HOST', 'localhost:9092')
video_convert_input_topic = os.environ.get('KAFKA_VIDEO_CONVERT_INPUT_TOPIC', 'video-convert-input-v1')
video_convert_output_topic = os.environ.get('KAFKA_VIDEO_CONVERT_OUTPUT_TOPIC', 'video-convert-output-v1')
video_convert_consumer_grp = os.environ.get('KAFKA_VIDEO_CONVERT_CONSUMER_GRP', 'video-convert-group')
total_no_of_partitions = 3

app_host = os.environ.get('VIDEO_CONVERT_HOST', '0.0.0.0')
app_port = os.environ.get('VIDEO_CONVERT_PORT', 5001)


