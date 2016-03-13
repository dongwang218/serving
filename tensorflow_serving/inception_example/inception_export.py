# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Export Inception model to tensorflow serving format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow_serving.session_bundle import exporter

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('export_path', '/tmp/inception_model',
                           """Absolute path to export dir.""")
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.Graph().as_default():
    with tf.Session() as sess:
      with tf.gfile.FastGFile(os.path.join(
          FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def export_model():
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  # Creates graph from saved GraphDef.
  #create_graph()

  print('Exporting trained model to ' + FLAGS.export_path)
  with tf.Graph().as_default():
    with tf.Session() as sess:
      with tf.gfile.FastGFile(os.path.join(
          FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

      # Some useful tensors:
      # 'softmax:0': A tensor containing the normalized prediction across
      #   1000 labels.
      # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
      #   float description of the image.
      # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
      #   encoding of the image.
      # Runs the softmax tensor by feeding the image_data as input to the graph.
      y = sess.graph.get_tensor_by_name('softmax:0')
      x = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
      yy = tf.Variable(1.0, name='yy')
      tf.initialize_all_variables().run()

      print(y)
      print(x)
      print(yy)
      for node in sess.graph.as_graph_def().node:
        print(node.name, node.op)
        
      saver = tf.train.Saver(sharded=True)
      model_exporter = exporter.Exporter(saver)
      signature = exporter.classification_signature(input_tensor=x, scores_tensor=y)
      model_exporter.init(sess.graph.as_graph_def(),
                          default_graph_signature=signature)
      model_exporter.export(FLAGS.export_path, tf.constant(FLAGS.export_version), sess)
      print('Done exporting!')

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  export_model()

if __name__ == '__main__':
  tf.app.run()
