# SPDX-License-Identifier: Apache-2.0

"""
A simple example how to call tensorflow-onnx via python.
"""

import tensorflow as tf
import tf2onnxnightly

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnxnightly.tfonnx.process_tf_graph(sess.graph, input_names=["input:0"], output_names=["output:0"])
    model_proto = onnx_graph.make_model("test")
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
