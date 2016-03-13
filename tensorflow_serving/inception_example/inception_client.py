# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/grte/v4/bin/python2.7

"""A client that talks to inception_inference service.

The client downloads test images of inception data set, queries the service with
such test images to get classification, and calculates the inference error rate.
Please see inception_inference.proto for details.

Typical usage example:

    inception_client.py --num_tests=100 --server=localhost:9000
"""
import os
import urllib
import sys
import threading
import re

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

import cv2
from tensorflow_serving.inception_example import inception_inference_pb2

sys.path.append('/home/ubuntu/caffe/python')

import caffe

tf.app.flags.DEFINE_string('server', '', 'inception_inference service host:port')
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

FLAGS = tf.app.flags.FLAGS


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


alldone = False

def do_inference(hostport, image):
  """Tests inception_inference service with concurrent requests.

  Args:
    hostport: Host:port address of the inception_inference service.
    concurrency: Maximum number of concurrent requests.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  
  # Creates node ID --> English string lookup.
  node_lookup = NodeLookup()

  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  cv = threading.Condition()

  def done(result_future):
    global alldone
    with cv:
      exception = result_future.exception()
      if exception:
        print exception
      else:
        predictions = numpy.array(result_future.result().value)
        top_k = predictions.argsort()[-5:][::-1]
        for node_id in top_k:
          human_string = node_lookup.id_to_string(node_id)
          score = predictions[node_id]
          print('%s (score = %.5f)' % (human_string, score))

      alldone = True
      cv.notify()
      
  request = inception_inference_pb2.InceptionRequest()
  request.image_data = image_data
  result_future = stub.Classify.future(request, 5.0)  # 5 seconds
  result_future.add_done_callback(
    lambda result_future: done(result_future))  # pylint: disable=cell-var-from-loop
  with cv:
    while alldone != True:
      cv.wait()

      
def main(_):
  if not FLAGS.server:
    print 'please specify server host:port'
    return

  if FLAGS.image_file.startswith('http'):
    image = '/tmp/' + FLAGS.image_file.split('/')[-1]
    urllib.urlretrieve(FLAGS.image_file, image)
  else:
    image = FLAGS.image_file

  do_inference(FLAGS.server, image)
if __name__ == '__main__':
  tf.app.run()
