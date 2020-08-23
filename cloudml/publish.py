# ensure that you are using virtualenv
# as described in the python dev setup guide

# pip install --upgrade google-cloud-pubsub
import json
from google.cloud import pubsub_v1
import os

project_id = "iwasnothing-self-learning"
topic_id = "submit_order_topic"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/kahingleung/Downloads/iwasnothing-self-learning-66b3ef91165a.json"
publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
topic_path = publisher.topic_path(project_id, topic_id)
j = {"ticker": "MDLZ", "spread": -0.03501995438868731}
data = json.dumps(j)
# Data must be a bytestring
data = data.encode("utf-8")
# When you publish a message, the client returns a future.
future = publisher.publish(topic_path, data=data)
print(future.result())

print("Published messages.")