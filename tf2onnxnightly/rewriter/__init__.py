# SPDX-License-Identifier: Apache-2.0

"""tf2onnx.rewriter module."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tf2onnxnightly.rewriter.cond_rewriter import rewrite_cond
from tf2onnxnightly.rewriter.conv2d_with_pad_rewriter import rewrite_conv2d_with_pad
from tf2onnxnightly.rewriter.dropout_rewriter import rewrite_dropout
from tf2onnxnightly.rewriter.eye_rewriter import rewrite_eye
from tf2onnxnightly.rewriter.flatten_rewriter import rewrite_flatten
from tf2onnxnightly.rewriter.gemm_rewriter import rewrite_gemm
from tf2onnxnightly.rewriter.leakyrelu_rewriter import rewrite_leakyrelu
from tf2onnxnightly.rewriter.random_normal_rewriter import rewrite_random_normal
from tf2onnxnightly.rewriter.random_uniform import rewrite_random_uniform, rewrite_random_uniform_fold_const
from tf2onnxnightly.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm, \
    rewrite_single_direction_gru, rewrite_bi_direction_gru, \
    rewrite_custom_rnn_cell, rewrite_generic_loop
from tf2onnxnightly.rewriter.thresholded_relu_rewriter import rewrite_thresholded_relu
from tf2onnxnightly.rewriter.transpose_rewriter import rewrite_transpose
from tf2onnxnightly.rewriter.conv2d_with_add_rewriter import rewrite_biasadd_with_conv2d
from tf2onnxnightly.rewriter.quantization_ops_rewriter import rewrite_quantize_and_dequantize


__all__ = [
    "rewrite_cond",
    "rewrite_conv2d_with_pad",
    "rewrite_dropout",
    "rewrite_eye",
    "rewrite_flatten",
    "rewrite_gemm",
    "rewrite_leakyrelu",
    "rewrite_random_normal",
    "rewrite_random_uniform",
    "rewrite_random_uniform_fold_const",
    "rewrite_thresholded_relu",
    "rewrite_transpose",
    "rewrite_single_direction_lstm",
    "rewrite_bi_direction_lstm",
    "rewrite_single_direction_gru",
    "rewrite_bi_direction_gru",
    "rewrite_custom_rnn_cell",
    "rewrite_generic_loop",
    "rewrite_biasadd_with_conv2d",
    "rewrite_quantize_and_dequantize"
]
