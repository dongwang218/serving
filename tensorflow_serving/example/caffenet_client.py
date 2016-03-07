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

"""A client that talks to mnist_inference service.

The client downloads test images of mnist data set, queries the service with
such test images to get classification, and calculates the inference error rate.
Please see mnist_inference.proto for details.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

import urllib
import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

import cv2
from tensorflow_serving.example import mnist_inference_pb2

sys.path.append('/home/ubuntu/caffe/python')

import caffe

tf.app.flags.DEFINE_string('server', '', 'mnist_inference service host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_string('label_file', '',
                           """Absolute path to ilsvrc12 label file.""")
FLAGS = tf.app.flags.FLAGS


batch_size=500
scale_size=256
crop_size=227
mean = numpy.array([104., 117., 124.])

def read_image(path):
        img = cv2.imread(path)
        h, w, c = numpy.shape(img)
        assert c==3
        resize_to = (scale_size, scale_size)
        img = cv2.resize(img, resize_to)
        img = img.astype(numpy.float32)
        img -= mean
        h, w, c = img.shape
        ho, wo = ((h-crop_size)/2, (w-crop_size)/2)
        img = img[ho:ho+crop_size, wo:wo+crop_size, :]
        img = img[None, ...]
        return img

alldone = False

def do_inference(hostport, work_dir, image, label_file):
  """Tests mnist_inference service with concurrent requests.

  Args:
    hostport: Host:port address of the mnist_inference service.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = read_image(image)
  # image_data = tf.gfile.FastGFile(image, 'rb').read()

  with open(label_file) as f:
    label = f.read().splitlines()

  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = mnist_inference_pb2.beta_create_MnistService_stub(channel)
  cv = threading.Condition()

  def done(result_future):
    global alldone
    with cv:
      exception = result_future.exception()
      if exception:
        print exception
      else:
        response = numpy.array(result_future.result().value)
        # print response
        top5 = (-response).argsort()[:5]
        for idx in top5:
          print response[idx], ':', label[idx]
      alldone = True
      cv.notify()
      
  request = mnist_inference_pb2.MnistRequest()
  for pixel in image_data.flatten():
    request.image_data.append(pixel.item())
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
  do_inference(FLAGS.server, FLAGS.work_dir,
                            image, FLAGS.label_file)
if __name__ == '__main__':
  tf.app.run()
