# SPDX-License-Identifier: Apache-2.0


"""reshape optimizer
   Finds reshape ops with computed shapes and attempts to replace them with constant shapes
"""

from .optimizer_base import GraphOptimizerBase
from .const_fold_optimizer import ConstFoldOptimizer
import numpy as np
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from collections import Counter

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring

class SymbolicShapeTensor:
    def __init__(self, shape, data):
        self.shape = shape
        self.data = data

    @staticmethod
    def from_list(shape):
        rank = [len(shape)]
        data = [SymbolicShapeTensorValue(i, v) for i, v in enumerate(shape)]
        return SymbolicShapeTensor(rank, data)

    def aslist(self):
        raise Exception("blah")

    def reshape(self, shape):
        return SymbolicShapeTensor(shape, self.data)


class SymbolicShapeTensorValue:
    def __init__(self, indices, value):
        self.indices = indices
        self.value = value
        if self.value == 0:
            self.indices = []

    def is_const(self):
        return len(self.indices) == 0

    def get_offset(self, i):
        if not self.is_idx():
            return None
        return self.indices[0] - i

    def get_reshape_dim(self, i, offset):
        if self.is_const():
            return self.value
        if self.get_offset(i) == offset:
            return 0
        return -1

    def is_one(self):
        return len(self.indices) == 0 and self.value == 1

    def is_idx(self):
        return len(self.indices) == 1 and self.value == 1

    def is_product(self):
        terms = len(self.indices)
        if self.value != 1:
            terms += 1
        return terms > 1

    def __mul__(self, other):
        if isinstance(other, SymbolicShapeTensorValue):
            return SymbolicShapeTensorValue(self.indices + other.indices, self.value * other.value)
        else:
            return SymbolicShapeTensorValue(self.indices, self.value * other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    @staticmethod
    def value_for_dim(idx, value):
        if value < 0:
            return SymbolicShapeTensorValue([idx], 1)
        else:
            return SymbolicShapeTensorValue([], value)

    @staticmethod
    def np_array_for_shape(shape):
        return np.array([SymbolicShapeTensorValue.value_for_dim(i, v) for i, v in enumerate(shape)], np.object)


class ReshapeOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(ReshapeOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if op.type == "Reshape" and self._optimize_reshape(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _optimize_reshape(self, node, graph):
        if node.inputs[1].is_const():
            return False
        nodes_to_compute_shape = self._trace_reshape_to_tensor(graph, node.inputs[1], node.input[0])
        if nodes_to_compute_shape is None:
            return False
        inp_shape = graph.get_shape(node.input[0])
        if inp_shape is None:
            inp_shape = [-1] * 3
            #return False
        symbolic_shape = self._symbolic_compute_shape(graph, nodes_to_compute_shape)
        # utils.make_sure(len(symbolic_shape.shape) == 1, "Shape must have rank 1")
        # symbolic_shape = symbolic_shape.tolist()
        product_cnt = len([val for val in symbolic_shape if val.is_product()])
        idx_cnt = len([val for val in symbolic_shape if val.is_idx()])
        if product_cnt > 1:
            return False
        if idx_cnt + product_cnt <= 1:
            new_shape = [v.value if v.is_const() else -1 for v in symbolic_shape]
            offset = 0
        else:
            offsets = [val.get_offset(i) for i, val in enumerate(symbolic_shape)]
            offset = Counter(o for o in offsets if o is not None).most_common(1)[0][0]
            new_shape = [v.get_reshape_dim(i, offset) for i, v in enumerate(symbolic_shape)]
        if new_shape.count(-1) > 1:
            return False

        if offset > 0:
            new_shape = [1] * offset + new_shape
            squeeze_node = GraphBuilder(graph).make_squeeze(
                {'data': node.output[0], 'axes': list(range(offset))},
                return_node=True, shapes=node.output_shapes, dtypes=node.output_dtypes)
            graph.insert_node_on_output(squeeze_node, node.output[0])
        const_shape = graph.make_const(utils.make_name(node.name + "_shape"), np.array(new_shape, np.int64)).output[0]
        graph.replace_inputs(node, [node.input[0], const_shape])
        if offset < 0:
            unsqueeze_node = GraphBuilder(graph).make_unsqueeze({'data': node.input[0], 'axes': list(range(-offset))})
            graph.replace_inputs(node, [unsqueeze_node, const_shape])

        return True

    def _make_plan(self, inp_shape, symbolic_shape):
        pass

    def _symbolic_compute_shape(self, graph, nodes_to_compute_shape):
        results = {}
        for node in nodes_to_compute_shape:
            if node.type == "Const":
                results[node.output[0]] = node.get_tensor_value(as_list=False)
            if node.type == "Shape":
                shape = graph.get_shape(node.input[0])
                if shape is None:
                    shape = [-1] * 3
                if -1 in shape:
                    results[node.output[0]] = SymbolicShapeTensorValue.np_array_for_shape(shape)
                else:
                    results[node.output[0]] = np.array(shape, np.int64)
            if node.type in ["Squeeze", "Unsqueeze"]:
                inp1 = results[node.input[0]]
                if graph.opset < 13:
                    axes = node.get_attr_value("axes")
                else:
                    axes = results[node.input[1]].tolist()
                shape = inp1.shape
                handler = self.compute_unsqueeze if node.type == "Unsqueeze" else self.compute_squeeze
                new_shape = handler(shape, axes)
                results[node.output[0]] = inp1.reshape(new_shape)
            if node.type == "Cast":
                inp = results[node.input[0]]
                if inp.dtype == np.object:
                    results[node.output[0]] = inp
                else:
                    np_dtype = utils.ONNX_TO_NUMPY_DTYPE[node.get_attr("to").i]
                    results[node.output[0]] = inp.astype(np_dtype)
            if node.type == "Mul":
                results[node.output[0]] = results[node.input[0]] * results[node.input[0]]
            if node.type == "ReduceProd":
                inp = results[node.input[0]]
                axes = node.get_attr_value("axes")
                keepdims = node.get_attr_value("keepdims", 1)
                results[node.output[0]] = np.prod(inp, axis=tuple(axes), keepdims=keepdims)
            if node.type == "Slice":
                inps = [results[inp] for inp in node.input]
                rank = len(inps[0].shape)
                if len(inps) == 3:
                    inps.append(list(range(rank)))
                if len(inps) == 4:
                    inps.append([1] * rank)
                slices = [slice(None, None, None) for _ in range(rank)]
                data, starts, ends, axes, steps = inps
                for axis, start, end, step in zip(axes, starts, ends, steps):
                    slices[axis] = slice(start, end, step)
                results[node.output[0]] = data[tuple(slices)]
            if node.type == "Concat":
                axis = node.get_attr_value("axis")
                inps = [results[inp] for inp in node.input]
                dtype = inps[0].dtype
                if any(inp.dtype == np.object for inp in inps):
                    dtype = np.object
                results[node.output[0]] = np.concatenate(inps, axis=axis)
            if node.type == "Gather":
                data = results[node.input[0]]
                indices = results[node.input[1]]
                assert indices.dtype != np.object
                axis = node.get_attr_value("axis", 0)
                results[node.output[0]] = np.take(data, indices, axis=axis)
            results[node.output[0]] = np.array(results[node.output[0]])
        res = results[node.output[0]].tolist()
        return [SymbolicShapeTensorValue([], v) if not isinstance(v, SymbolicShapeTensorValue) else v for v in res]


    def compute_unsqueeze(self, shape_in, axes):
        dims_out = len(shape_in) + len(axes)
        axes = [i if i >= 0 else i + dims_out for i in axes]
        shape_in = iter(shape_in)
        shape_out = [None] * dims_out
        for ind in axes:
            shape_out[ind] = 1
        for ind, val in enumerate(shape_out):
            if val is None:
                shape_out[ind] = next(shape_in)
        return shape_out


    def compute_squeeze(self, shape_in, axes):
        axes = [i if i >= 0 else i + len(axes) for i in axes]
        shape_out = []
        for ind, val in enumerate(shape_in):
            if ind not in axes:
                shape_out.append(val)
        return shape_out
                

    def _trace_reshape_to_tensor(self, graph, node, output):
        inputs = []
        inputs += node.inputs
        compatible_types = { "Unsqueeze", "Squeeze", "Gather", "Mul", "ReduceProd", "Slice", "Cast", "Shape", "Concat", "Const" }
        sorted_nodes = []
        sorted_nodes.append(node)
        while inputs:
            inp = inputs.pop()
            if inp.type not in compatible_types:
                return None
            if inp.type == "Shape":
                if inp.input[0] != output:
                    return None
                sorted_nodes.append(inp)
            else:
                inputs += inp.inputs
                sorted_nodes.append(inp)
        return sorted_nodes[::-1]
        