# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewrite - rewrite tensorflow tensordot subgraph to onnx einsum op
"""
import logging
from onnx import onnx_pb
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring

def rewrite_tensordot(g, ops):
    if g.opset <= 11:
        return ops

    pattern0 = \
        OpTypePattern('Reshape', name='reshape_out', inputs=[
            OpTypePattern('MatMul', name='matmul', inputs=[
                OpTypePattern('Reshape', name='reshape_in1', inputs=[
                    OpTypePattern('Transpose|Identity', name='transpose1'),
                    OpTypePattern('Pack', inputs=[
                        OpTypePattern('Prod', inputs=[
                            OpTypePattern('GatherV2', name='gather1a', inputs=[
                                OpTypePattern('Shape', name='shape1r1'),
                                "Const|ConstV2",
                                "*"
                            ]),
                            "*"
                        ]),
                        OpTypePattern('Prod', inputs=[
                            OpTypePattern('GatherV2', name='gather1b', inputs=[
                                OpTypePattern('Shape', name='shape1r2'),
                                "Const|ConstV2",
                                "*"
                            ]),
                            "*"
                        ]),
                    ])
                ]),
                OpTypePattern('Reshape', name='reshape_in2', inputs=[
                    OpTypePattern('Transpose|Identity', name='transpose2'),
                    OpTypePattern('Pack', inputs=[
                        OpTypePattern('Prod', inputs=[
                            OpTypePattern('GatherV2', name='gather2a', inputs=[
                                OpTypePattern('Shape', name='shape2r1'),
                                "*",
                                "*"
                            ]),
                            "*"
                        ]),
                        OpTypePattern('Prod', inputs=[
                            OpTypePattern('GatherV2', name='gather2b', inputs=[
                                OpTypePattern('Shape', name='shape2r2'),
                                "*",
                                "*"
                            ]),
                            "*"
                        ]),
                    ])
                ]),
            ]),
            OpTypePattern('ConcatV2', name='concat', inputs=[
                OpTypePattern('GatherV2', name='gather1ar2'),
                OpTypePattern('GatherV2', name='gather2br2'),
                "*"
            ])
        ])

    pattern_list = [pattern0]

    for pattern in pattern_list:
        matcher = GraphMatcher(pattern, allow_reorder=False)
        match_results = list(matcher.match_ops(ops))
        if match_results:
            for match in match_results:

                inp1 = match.get_op("shape1r1").input[0]
                inp2 = match.get_op("shape2r1").input[0]
                reshape_out = match.get_op("reshape_out")

                gather1a = match.get_op("gather1a")
                gather1b = match.get_op("gather1b")
                gather2a = match.get_op("gather2a")
                gather2b = match.get_op("gather2b")

                if gather2b.name != match.get_op("gather2br2").name:
                    continue
                if gather1a.name != match.get_op("gather1ar2").name:
                    continue
                if match.get_op("shape1r1").name != match.get_op("shape1r2").name:
                    continue
                if match.get_op("shape2r1").name != match.get_op("shape2r2").name:
                    continue

                gather1a_idx = gather1a.inputs[1].get_tensor_value(as_list=True)
                gather1b_idx = gather1b.inputs[1].get_tensor_value(as_list=True)
                gather2a_idx = gather2a.inputs[1].get_tensor_value(as_list=True)
                gather2b_idx = gather2b.inputs[1].get_tensor_value(as_list=True)

                if len(gather1b_idx) != len(gather2a_idx) or len(gather1a_idx + gather1b_idx + gather2b_idx) > 26:
                    continue

                inp1_to_term = {}
                inp2_to_term = {}
                out_to_term = []

                term_cnt = [0]
                def make_term():
                    term = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[term_cnt[0]]
                    term_cnt[0] += 1
                    return term

                for inp1_dim in gather1a_idx:
                    t = make_term()
                    inp1_to_term[inp1_dim] = t
                    out_to_term.append(t)

                for inp1_dim, inp2_dim in zip(gather1b_idx, gather2a_idx):
                    t = make_term()
                    inp1_to_term[inp1_dim] = t
                    inp2_to_term[inp2_dim] = t
                
                for inp2_dim in gather2b_idx:
                    t = make_term()
                    inp2_to_term[inp2_dim] = t
                    out_to_term.append(t)

                term1 = "".join(v for k, v in sorted(inp1_to_term.items()))
                term2 = "".join(v for k, v in sorted(inp2_to_term.items()))
                term_out = "".join(v for v in out_to_term)
                equation = term1 + "," + term2 + "->" + term_out

                name = reshape_out.name
                outputs = reshape_out.output
                shapes = reshape_out.output_shapes
                dtypes = reshape_out.output_dtypes
                
                g.remove_node(reshape_out.name)
                einsum = g.make_node("Einsum", [inp1, inp2], attr={'equation': equation},
                                     outputs=outputs, shapes=shapes, dtypes=dtypes, name=name)
                ops.append(einsum)
    return ops


