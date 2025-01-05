# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# test cases are modified from tvm test_forward.py https://github.com/apache/tvm/blob/main/tests/python/frontend/onnx/test_forward.py

import glob
import os
import platform
import re
import copy
import tempfile
import pytest
import scipy
import numpy as np

# import tvm
# import tvm.testing
# import tvm.topi.testing
# from tvm import relay
# from tvm.contrib import graph_executor, utils
# from tvm.relay.frontend.common import infer_type
# from tvm.relay.build_module import bind_params_by_name
# from relay.utils.tag_span import _create_span, _set_span, _verify_structural_equal_with_span

import onnx
import onnxruntime.backend
from onnx import TensorProto, helper, mapping, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, quantize_static

import torch
import torchvision
from torch.nn import Linear, Module, Sequential


def get_input_data_shape_dict(graph_def, input_data):
    """Get input data shape"""
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            input_ = input_data[i]

            if input_ is None or not hasattr(input_, "shape") or input_.shape == ():
                # Skip adding input shape data when the input data is None;
                # This is to enable optional arguments for onnx operators.
                continue

            elif isinstance(input_, list):
                shape_dict[input_names[i]] = (len(input_),)

            else:
                shape_dict[input_names[i]] = input_.shape

    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict


def get_tinygrad_output_with_vm(
    graph_def,
    input_data,
    target,
    dev,
    opset=None,
    freeze_params=False,
    convert_config=None,
    validate_structural_equal=True,
):
    """Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_onnx(
            graph_def,
            shape_dict,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
        # handle the bfloat16 so we explicitly allocate
        # bfloat16 arrays as input
        for i, param in enumerate(mod["main"].params):
            if param.type_annotation.dtype == "bfloat16":
                input_data[i] = tvm.nd.empty(input_data[i].shape, "bfloat16").copyfrom(
                    input_data[i]
                )

    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_onnx(
                graph_def,
                shape_dict,
                opset=opset,
                freeze_params=freeze_params,
                convert_config=convert_config,
            )
        tvm.ir.assert_structural_equal(mod, mod_with_span)

    result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
        *input_data, **params
    )
    if isinstance(result, tvm.runtime.NDArray):
        return result.numpy()
    return [r.numpy() for r in result]


def get_tinygrad_output(
    graph_def,
    input_data,
    target,
    dev,
    output_shape=None,
    output_dtype="float32",
    opset=None,
    opt_level=1,
    convert_config=None,
):
    """Generic function to execute and get tvm output"""
    # TODO: Resolve the issues and remove the following lines
    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(
        graph_def, shape_dict, opset=opset, convert_config=convert_config
    )

    with tvm.transform.PassContext(opt_level=opt_level):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_executor.create(graph, lib, dev)
    # set inputs
    if isinstance(input_data, list):
        for i, _ in enumerate(input_names):
            # Its possible for some onnx inputs to not be needed in the tvm
            # module, confirm its present before setting.
            # pylint: disable=unnecessary-list-index-lookup
            m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.numpy()


def get_onnxruntime_output(model, inputs):
    """Generic function to generate onnxruntime output"""
    rep = onnxruntime.backend.prepare(model.SerializeToString(), "CPU")
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    output = rep.run(inp)
    # Unpack output if there's only a single value.
    if len(output) == 1:
        output = output[0]
    return output


def verify_with_ort_with_inputs(
    model,
    inputs,
    out_shape=None,
    target=None,
    dev=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
    apply_softmax=False,
    opt_level=1,
    convert_config=None,
):
    """verify_with_ort_with_inputs"""
    if opset is not None:
        model.opset_import[0].version = opset

    ort_out = get_onnxruntime_output(model, inputs)
    if use_vm:
        tvm_out = get_tinygrad_output_with_vm(
            model,
            inputs,
            target,
            dev,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
    else:
        tvm_out = get_tinygrad_output(
            model,
            inputs,
            target,
            dev,
            out_shape,
            dtype,
            opset=opset,
            opt_level=opt_level,
            convert_config=convert_config,
        )

    if not isinstance(tvm_out, list):
        tvm_out = [tvm_out]
    if not isinstance(ort_out, list):
        ort_out = [ort_out]
    for tvm_val, ort_val in zip(tvm_out, ort_out):
        if apply_softmax:
            ort_val = scipy.special.softmax(ort_val)
            tvm_val = scipy.special.softmax(tvm_val)
        tvm.testing.assert_allclose(ort_val, tvm_val, rtol=rtol, atol=atol)
        assert ort_val.dtype == tvm_val.dtype


def verify_with_ort(
    model,
    input_shapes,
    out_shape=None,
    target=None,
    dev=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
):
    """verify_with_ort"""
    inputs = [np.random.uniform(size=ishape).astype(dtype) for ishape in input_shapes]
    verify_with_ort_with_inputs(
        model,
        inputs,
        out_shape=out_shape,
        target=target,
        dev=dev,
        use_vm=use_vm,
        opset=opset,
        freeze_params=freeze_params,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )


def quantize_and_verify_with_ort(
    onnx_model, input_names, input_shapes, target, dev, rtol=1e-5, atol=1e-5
):
    """quantize_and_verify_with_ort"""
    input_arrays = [np.random.random(shape).astype("float32") for shape in input_shapes]

    class RandomDataReader(CalibrationDataReader):
        # pylint: disable=missing-class-docstring
        def __init__(self, n=10):
            input_dict = dict(zip(input_names, input_shapes))
            self.data = iter(
                [
                    {
                        name: np.random.random(shape).astype("float32")
                        for name, shape in input_dict.items()
                    }
                    for _ in range(n)
                ]
            )

        def get_next(self):
            return next(self.data, None)

    t_dir = tvm.contrib.utils.tempdir()
    model_fp32 = os.path.join(t_dir.temp_dir, "model.onnx")
    onnx.save_model(onnx_model, model_fp32)
    model_quant = os.path.join(t_dir.temp_dir, "model.quant.onnx")
    _ = quantize_static(  # pylint: disable=assignment-from-no-return
        model_fp32, model_quant, RandomDataReader()
    )
    # opt_level=1 will cause error with qnn lowering
    model = onnx.load(model_quant)
    verify_with_ort_with_inputs(
        model, input_arrays, opt_level=2, target=target, dev=dev, use_vm=True, rtol=rtol, atol=atol
    )


def make_constant_node(name, data_type, dims, vals):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )


def is_version_greater_than(ver):
    return "".join(re.findall(r"(\d+\.)(\d+\.)(\d)", onnx.__version__)[0]) > "".join(
        re.findall(r"(\d+\.)(\d+\.)(\d)", ver)[0]
    )


@tvm.testing.parametrize_targets
def test_reshape(target, dev):
    """test_reshape"""
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ref_in"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT32,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(int),
        ),
    )
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    graph = helper.make_graph(
        [ref_node, reshape_node],
        "reshape_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="reshape_test")

    x = np.random.uniform(size=in_shape).astype("int32")
    tvm_out = get_tinygrad_output(model, x, target, dev, ref_shape, "float32")
    tvm.testing.assert_allclose(ref_shape, tvm_out.shape)


@tvm.testing.parametrize_targets
def test_double_reshape(target, dev):
    """test_double_reshape"""
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ref_in"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT32,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(int),
        ),
    )
    reshape_node1 = helper.make_node("Reshape", ["in", "ref_in"], ["out1"])
    reshape_node2 = helper.make_node("Reshape", ["in", "ref_in"], ["out2"])
    add_node = helper.make_node("Add", ["out1", "out2"], ["out"])

    graph = helper.make_graph(
        [ref_node, reshape_node1, reshape_node2, add_node],
        "reshape_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="reshape_test")

    x = np.random.uniform(size=in_shape).astype("int32")
    tvm_out = get_tinygrad_output(model, x, target, dev, ref_shape, "float32")
    tvm.testing.assert_allclose(ref_shape, tvm_out.shape)


@tvm.testing.parametrize_targets
def test_expand(target, dev):
    """test_expand"""

    def _test_expand(name, data, shape, ref_data, dtype="int32"):
        shape_array = np.array(shape)
        if dtype == "int32":
            shape_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["shape"],
                value=onnx.helper.make_tensor(
                    name="const_tensor",
                    data_type=onnx.TensorProto.INT32,
                    dims=shape_array.shape,
                    vals=shape_array.flatten().astype("int32"),
                ),
            )
        elif dtype == "int64":
            shape_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["shape"],
                value=onnx.helper.make_tensor(
                    name="const_tensor",
                    data_type=onnx.TensorProto.INT64,
                    dims=shape_array.shape,
                    vals=shape_array.flatten().astype("int64"),
                ),
            )
        else:
            raise TypeError("Invalid dtype")
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_data.shape))],
        )

        model = helper.make_model(graph, producer_name=name)

        tvm_out = get_tinygrad_output_with_vm(model, data, target, dev, freeze_params=True)
        tvm.testing.assert_allclose(ref_data, tvm_out)

    in_shape = (3, 1)
    shape = (3, 4)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = np.tile(data, 4)
    _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data, "int32")
    _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data, "int64")

    in_shape = (3, 1)
    shape = (2, 1, 6)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = data * np.ones(shape, dtype=np.float32)
    _test_expand("expand_larger_target_shape_test", data, shape, ref_data, "int32")
    _test_expand("expand_larger_target_shape_test", data, shape, ref_data, "int64")

    in_shape = (1, 1)
    shape = (3,)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = data * np.ones(shape, dtype=np.float32)
    _test_expand("expand_smaller_target_shape_test", data, shape, ref_data, "int32")
    _test_expand("expand_smaller_target_shape_test", data, shape, ref_data, "int64")


@tvm.testing.parametrize_targets
def test_depth_to_space(target, dev):
    """test_depth_to_space"""

    def verify_depth_to_space(inshape, outshape, mode, block_size):
        node = onnx.helper.make_node(
            "DepthToSpace", inputs=["x"], outputs=["y"], blocksize=block_size
        )

        graph = helper.make_graph(
            [node],
            "depth_to_space_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
        )

        model = helper.make_model(graph, producer_name="depth_to_space_test")

        verify_with_ort(model, [inshape], [outshape], target, dev)

    # current onnx.checker use OpSet-1 version of DepthToSpace, which doesn't have a mode argument.
    # TO-DO, we can add mode argument to test CRD mode and DCR mode
    # in the future when we update to a newer onnx version.
    verify_depth_to_space((1, 8, 2, 3), (1, 2, 4, 6), mode="CRD", block_size=2)


@tvm.testing.parametrize_targets
def test_space_to_depth(target, dev):
    """test_space_to_depth"""

    def verify_space_to_depth(inshape, outshape, block_size):
        node = onnx.helper.make_node(
            "SpaceToDepth", inputs=["x"], outputs=["y"], blocksize=block_size
        )

        graph = helper.make_graph(
            [node],
            "space_to_depth_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
        )

        model = helper.make_model(graph, producer_name="space_to_depth_test")

        verify_with_ort(model, [inshape], [outshape], target, dev)

    verify_space_to_depth((1, 1, 4, 6), (1, 4, 2, 3), 2)


@tvm.testing.parametrize_targets
def test_shape(target, dev):
    """test_shape"""
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ref_in"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT32,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(int),
        ),
    )
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    shape_node = helper.make_node("Shape", ["out"], ["final_out"])

    graph = helper.make_graph(
        [ref_node, reshape_node, shape_node],
        "shape_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("final_out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="shape_test")

    x = np.random.uniform(size=in_shape).astype("int32")
    tvm_out = get_tinygrad_output(model, x, target, dev, ref_shape, "int32")
    tvm.testing.assert_allclose(ref_shape, tvm_out)


@tvm.testing.parametrize_targets
def test_power(target, dev):
    """test_power"""

    def _test_power_iteration(x_shape, y_shape):
        if isinstance(y_shape, int):
            y_shape = [y_shape]

        x = np.random.uniform(size=x_shape).astype(np.float32)
        y = np.random.uniform(size=y_shape).astype(np.float32)

        np_res = np.power(x, y).astype(np.float32)

        res = helper.make_node("Pow", ["x", "y"], ["out"])

        graph = helper.make_graph(
            [res],
            "power_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(np_res.shape))],
        )

        model = helper.make_model(graph, producer_name="power_test")

        tvm_out = get_tinygrad_output(model, [x, y], target, dev, np_res.shape)
        tvm.testing.assert_allclose(np_res, tvm_out, rtol=1e-5, atol=1e-5)

    _test_power_iteration((1, 3), (1))
    _test_power_iteration((2, 3), (2, 3))
    _test_power_iteration((2, 3), (1, 3))


@tvm.testing.parametrize_targets
def test_range(target, dev):
    """test_range"""

    def verify_range(start, limit, delta, dtype):
        dtype_map = {
            "float32": TensorProto.FLOAT,
            "int32": TensorProto.INT32,
            "int64": TensorProto.INT64,
        }
        dtype_onnx = dtype_map[dtype]
        y = helper.make_node("Range", ["start", "limit", "delta"], ["output"])
        graph = helper.make_graph(
            [y],
            "range_test",
            inputs=[
                helper.make_tensor_value_info("start", dtype_onnx, []),
                helper.make_tensor_value_info("limit", dtype_onnx, []),
                helper.make_tensor_value_info("delta", dtype_onnx, []),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", dtype_onnx, np.arange(start, limit, delta).shape
                )
            ],
        )
        model = helper.make_model(graph, producer_name="range_test")
        inputs = [np.array(x).astype(dtype) for x in [start, limit, delta]]
        verify_with_ort_with_inputs(model, inputs, target=target, dev=dev, use_vm=True)

    for t in ["float32", "int32", "int64"]:
        verify_range(0, 10, 1, t)
        verify_range(2, 8, 2, t)
        verify_range(-3, 6, 4, t)
        verify_range(-2, -7, -1, t)


@tvm.testing.parametrize_targets
def test_squeeze(target, dev):
    """test_squeeze"""

    def test_squeeze_once(in_shape, out_shape, axes=None):
        y = helper.make_node("Squeeze", ["in"], ["out"], axes=axes)

        graph = helper.make_graph(
            [y],
            "squeeze_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="squeeze_test")
        x = np.random.uniform(size=in_shape).astype("float32")
        verify_with_ort_with_inputs(model, [x], [out_shape], target=target, dev=dev, opset=11)

    test_squeeze_once((1, 3, 1, 3, 1, 1), (3, 3), [0, 2, 4, 5])
    test_squeeze_once((1, 3, 1, 3, 1, 1), (3, 3))  # empty axis.
    test_squeeze_once((), ())  # scalar testing.


@tvm.testing.parametrize_targets
def test_flatten(target, dev):
    """test_flatten"""

    def verify_flatten(in_shape, axis, ref_shape):
        flatten_node = helper.make_node("Flatten", ["in"], ["out"], axis=axis)

        graph = helper.make_graph(
            [flatten_node],
            "flatten_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))],
        )

        model = helper.make_model(graph, producer_name="flatten_test")
        verify_with_ort(model, [in_shape], target=target, dev=dev)

    verify_flatten((1, 3, 4, 4), 1, (1, 48))
    verify_flatten((1,), 1, (1, 1))


@tvm.testing.parametrize_targets
def test_unsqueeze(target, dev):
    """test_unsqueeze"""
    in_shape = (3, 3)
    axis = (0, 3, 4)
    out_shape = (1, 3, 3, 1, 1)
    y = helper.make_node("Unsqueeze", ["in"], ["out"], axes=list(axis))

    graph = helper.make_graph(
        [y],
        "squeeze_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    verify_with_ort(model, [in_shape], target=target, dev=dev, opset=11)


@tvm.testing.parametrize_targets
def test_unsqueeze_with_neg_axes(target, dev):
    def verify_unsqueeze_with_neg_axes(opset=11):
        in_shape = (2, 3, 4)
        axis = (-2, -1)
        out_shape = (2, 3, 4, 1, 1)
        if opset < 13:
            y = helper.make_node("Unsqueeze", ["in"], ["out"], axes=list(axis))
            nodes = [y]
        else:
            axes = np.array(list(axis)).astype(np.int64)
            axes = helper.make_node(
                "Constant",
                inputs=[],
                outputs=["axes"],
                value=onnx.helper.make_tensor(
                    name="const_axes",
                    data_type=onnx.TensorProto.INT64,
                    dims=axes.shape,
                    vals=axes.flatten().astype(int),
                ),
            )
            y = helper.make_node("Unsqueeze", ["in", "axes"], ["out"])
            nodes = [axes, y]

        graph = helper.make_graph(
            nodes,
            "squeeze_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="squeeze_test")
        verify_with_ort(model, [in_shape], target=target, dev=dev, opset=opset)

    verify_unsqueeze_with_neg_axes()
    verify_unsqueeze_with_neg_axes(opset=13)


@tvm.testing.parametrize_targets
def test_gather(target, dev):
    """test_gather"""

    def verify_gather(in_shape, indices, axis, dtype):
        x = np.random.uniform(size=in_shape).astype(dtype)
        indices = np.array(indices, dtype="int64")
        out_np = np.take(x, indices, axis=axis)

        y = helper.make_node("Gather", ["in", "indices"], ["out"], axis=axis)

        graph = helper.make_graph(
            [y],
            "gather_test",
            inputs=[
                helper.make_tensor_value_info(
                    "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(in_shape)
                ),
                helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(out_np.shape)
                )
            ],
        )
        model = helper.make_model(graph, producer_name="gather_test")
        verify_with_ort_with_inputs(model, [x, indices], target=target, dev=dev, dtype=dtype)

    verify_gather((4,), [1], 0, "int32")
    verify_gather((1, 4), [0], 0, "int32")
    verify_gather((4,), [[[1, 0], [0, 1]]], 0, "float32")
    verify_gather((2, 2), [[[1, 0], [0, 1]]], 1, "int32")
    verify_gather((3, 3, 3), [[[1, 0]]], -1, "int32")
    verify_gather((4, 3, 5, 6), [[2, 1, 0, 0]], 0, "float32")


@tvm.testing.parametrize_targets
def test_dynamic_gather(target, dev):
    """test_dynamic_gather"""
    dtype = "float32"
    in_shape = [2, 2]
    indices = 1
    axis = 1
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int64")
    out_np = np.take(x, indices, axis=axis)

    indices = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["indices"],
        value=onnx.helper.make_tensor(
            name="const_indices",
            data_type=onnx.TensorProto.INT64,
            dims=[],
            vals=[1],
        ),
    )
    y = helper.make_node("Gather", ["in", "indices"], ["out"], axis=axis)

    graph = helper.make_graph(
        [indices, y],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info(
                "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], ["?", "?"]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], ["?"] * len(out_np.shape)
            )
        ],
    )
    model = helper.make_model(graph, producer_name="dynamic_gather_test")

    mod, params = relay.frontend.from_onnx(model)

    result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(x, **params)
    tvm.testing.assert_allclose(out_np, result.numpy(), rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_gatherelements(target, dev):
    """test_gatherelements"""

    def verify_gatherelements(in_shape, indices, axis):
        x = np.random.uniform(size=in_shape).astype("float32")
        indices = np.array(indices, dtype="int32")

        y = helper.make_node("GatherElements", ["data", "indices"], ["output"], axis=axis)
        graph = helper.make_graph(
            [y],
            "gather_elements_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(in_shape))],
        )
        model = helper.make_model(graph, producer_name="gather_elements_test")

        verify_with_ort_with_inputs(model, [x, indices], target=target, dev=dev)

    verify_gatherelements((4,), [3, 0, 2, 1], 0)
    verify_gatherelements((2, 2), [[1, 0], [0, 1]], 0)
    verify_gatherelements((2, 2), [[0, 0], [1, 0]], 1)
    verify_gatherelements((2, 2), [[1, 0], [0, 1]], 1)

    indices = [
        [[1, 0, 0], [1, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 2, 1], [1, 0, 1]],
        [[1, 2, 1], [1, 2, 1], [1, 2, 1]],
    ]

    verify_gatherelements((3, 3, 3), indices, 2)


@tvm.testing.parametrize_targets
def test_scatter(target, dev):
    """test_scatter"""

    def verify_scatter(in_shape, indices, axis):
        x = np.random.uniform(size=in_shape).astype("float32")
        indices = np.array(indices, dtype="int32")
        updates = np.random.uniform(size=indices.shape).astype("float32")

        y = helper.make_node("Scatter", ["data", "indices", "updates"], ["output"], axis=axis)

        graph = helper.make_graph(
            [y],
            "scatter_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(indices.shape)),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(in_shape))],
        )
        model = helper.make_model(graph, producer_name="scatter_test")
        # Scatter operator has been supported from version 9 and
        # deprecated since version 11 of the default ONNX operator set
        verify_with_ort_with_inputs(model, [x, indices, updates], target=target, dev=dev, opset=9)

    verify_scatter((4,), [1], 0)
    verify_scatter((1, 4), [[0]], 0)
    verify_scatter((4,), [2, 3], 0)
    verify_scatter((2, 2), [[1, 0], [0, 1]], 1)
    verify_scatter((3, 3, 3), [[[-1, -3]]], -1)
    verify_scatter((4, 3, 5, 6), [[[[2, 1, 0, 0]]]], 0)


@tvm.testing.parametrize_targets
def test_scatter_elements(target, dev):
    """test_scatter_elements"""

    def verify_scatter_elements(in_shape, indices, axis=0, reduction="update"):
        x = np.random.uniform(size=in_shape).astype("float32")
        indices = np.array(indices, dtype="int32")
        updates = np.random.uniform(size=indices.shape).astype("float32")

        scatter_elements_node = helper.make_node(
            "ScatterElements",
            ["data", "indices", "updates"],
            ["output"],
            axis=axis,
            reduction=reduction,
        )

        graph = helper.make_graph(
            [scatter_elements_node],
            "scatter_elements_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(indices.shape)),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(in_shape))],
        )
        model = helper.make_model(graph, producer_name="scatter_elements_test")
        verify_with_ort_with_inputs(model, [x, indices, updates], target=target, dev=dev)

    # Usual scatter for 1d input
    verify_scatter_elements((4,), [2, 3])
    # Usual scatter with specified positive axis
    verify_scatter_elements((2, 2), [[1, 0], [0, 1]], 1)
    # Usual scatter for 3d input with spicified negative indices and axis
    verify_scatter_elements((3, 3, 3), [[[-1, -3]]], -1)
    # Usual scatter for 4d input
    verify_scatter_elements((4, 3, 5, 6), [[[[2, 1, 0, 0]]]])
    # Scatter elements with addition reduction of duplicates
    verify_scatter_elements(
        (3, 3, 3),
        [[[0, 2, 1], [1, 1, 1], [2, 1, 0]], [[0, 2, 1], [1, 1, 1], [2, 1, 0]]],
        0,
        "add",
    )
    # Scatter elements with reduction and specified axis
    verify_scatter_elements((3, 3, 3), [[[2, 2, 2], [1, 1, 1], [0, 0, 0]]], 2, "add")
    # Scatter elements with multiplication reduction of duplicates
    verify_scatter_elements(
        (3, 3, 3),
        [[[0, 2, 1], [1, 1, 1], [2, 1, 0]], [[0, 2, 1], [1, 1, 1], [2, 1, 0]]],
        0,
        "mul",
    )
    # TODO(vvchernov): min and max options are supported from 18 version, but CI supports 17 only
    # # Scatter elements with min reduction of duplicates
    # verify_scatter_elements(
    #     (3, 3, 3),
    #     [[[0, 2, 1], [1, 1, 1], [2, 1, 0]], [[0, 2, 1], [1, 1, 1], [2, 1, 0]]],
    #     0,
    #     "min",
    # )
    # # Scatter elements with max reduction of duplicates
    # verify_scatter_elements(
    #     (3, 3, 3),
    #     [[[0, 2, 1], [1, 1, 1], [2, 1, 0]], [[0, 2, 1], [1, 1, 1], [2, 1, 0]]],
    #     0,
    #     "max",
    # )


@tvm.testing.parametrize_targets
def test_slice(target, dev):
    """test_slice"""

    def _test_slice_iteration_v1(indata, outdata, starts, ends, axes=None):
        if axes:
            y = helper.make_node("Slice", ["in"], ["out"], axes=axes, starts=starts, ends=ends)
        else:
            y = helper.make_node("Slice", ["in"], ["out"], starts=starts, ends=ends)

        graph = helper.make_graph(
            [y],
            "slice_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="slice_test")
        verify_with_ort_with_inputs(
            model, [indata], [outdata.shape], opset=1, target=target, dev=dev
        )

    def _test_slice_iteration_v10(indata, outdata, **attrs):
        starts = attrs["starts"]
        ends = attrs["ends"]
        axes = None if "axes" not in attrs else attrs["axes"]
        steps = None if "steps" not in attrs else attrs["steps"]
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        inputs = [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(indata.shape)),
            helper.make_tensor_value_info("starts", TensorProto.INT64, list(starts.shape)),
            helper.make_tensor_value_info("ends", TensorProto.INT64, list(ends.shape)),
        ]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, list(starts.shape), starts),
            helper.make_tensor("ends", TensorProto.INT64, list(ends.shape), ends),
        ]
        nodes = []

        if "add_noop_to_input_attrs" in attrs:

            def add_noop_to_input_attr(attr_name, attr):
                output_name = attr_name + "_output"

                ref_shape = list(np.array(attr).shape)
                ref_shape.insert(0, 1)
                ref_shape = tuple(ref_shape)
                ref_array = np.array(ref_shape)
                ref_node = onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["ref_in_" + attr_name],
                    value=onnx.helper.make_tensor(
                        name="const_tensor__1_" + attr_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=ref_array.shape,
                        vals=ref_array.flatten().astype(int),
                    ),
                )
                in_shape = np.array(attr).shape
                in_array = np.array(in_shape)
                ref_node2 = onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["input_shape_" + attr_name],
                    value=onnx.helper.make_tensor(
                        name="const_tensor__2_" + attr_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=in_array.shape,
                        vals=in_array.flatten().astype(int),
                    ),
                )

                reshape1_node = helper.make_node(
                    "Reshape", [attr_name, "ref_in_" + attr_name], ["reshape_" + attr_name]
                )
                reshape2_node = helper.make_node(
                    "Reshape", ["reshape_" + attr_name, "input_shape_" + attr_name], [output_name]
                )
                return [ref_node, ref_node2, reshape1_node, reshape2_node]

        slice_inputs = []
        for attr_name in ["starts", "ends", "axes", "steps"]:
            if attr_name not in attrs:
                continue
            if "add_noop_to_input_attrs" in attrs and attr_name in attrs["add_noop_to_input_attrs"]:
                nodes.extend(add_noop_to_input_attr(attr_name, attrs[attr_name]))
                slice_inputs.append(attr_name + "_output")
            else:
                slice_inputs.append(attr_name)

        if axes:
            axes = np.asarray(axes)
            inputs.append(
                helper.make_tensor_value_info("axes", TensorProto.INT64, list(axes.shape))
            )
            initializer.append(
                helper.make_tensor("axes", TensorProto.INT64, list(axes.shape), axes)
            )

        if steps:
            assert axes is not None and len(axes) == len(steps)
            steps = np.asarray(steps)
            inputs.append(
                helper.make_tensor_value_info("steps", TensorProto.INT64, list(axes.shape))
            )
            initializer.append(
                helper.make_tensor("steps", TensorProto.INT64, list(steps.shape), steps)
            )

        y = helper.make_node("Slice", ["data", *slice_inputs], ["out"])

        nodes.append(y)
        graph = helper.make_graph(
            nodes,
            "slice_test",
            inputs=inputs,
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
            initializer=initializer,
        )
        model = helper.make_model(graph, producer_name="slice_test")
        verify_with_ort_with_inputs(
            model, [indata], opset=10, freeze_params=True, use_vm=True, target=target, dev=dev
        )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    _test_slice_iteration_v1(x, x[0:3, 0:10], starts=(0, 0), ends=(3, 10), axes=(0, 1))
    _test_slice_iteration_v1(x, x[0:3, 0:10], starts=(0, 0), ends=(10, 3), axes=(1, 0))
    _test_slice_iteration_v1(x, x[:, :, 3:4], starts=(0, 0, 3), ends=(20, 10, 4))
    _test_slice_iteration_v1(x, x[:, 1:1000], starts=(1,), ends=(1000,), axes=(1,))
    _test_slice_iteration_v1(x, x[:, 0:-1], starts=(0,), ends=(-1,), axes=(1,))
    _test_slice_iteration_v10(x, x[0:3, 0:10], starts=(0, 0), ends=(3, 10), axes=(0, 1))
    _test_slice_iteration_v10(x, x[0:3, 0:10], starts=(0, 0), ends=(10, 3), axes=(1, 0))
    _test_slice_iteration_v10(x, x[:, :, 3:4], starts=(0, 0, 3), ends=(20, 10, 4))
    _test_slice_iteration_v10(x, x[:, 1:1000], starts=(1,), ends=(1000,), axes=(1,))
    _test_slice_iteration_v10(x, x[:, 0:-1], starts=(0,), ends=(-1,), axes=(1,))
    _test_slice_iteration_v10(x, x[:, 0:-1], starts=(0,), ends=(-1,), axes=(-1,))
    _test_slice_iteration_v10(
        x,
        x[0:3, 0:10],
        starts=(0, 0),
        ends=(3, 10),
        axes=(0, 1),
        add_noop_to_input_attrs=["starts"],
    )
    _test_slice_iteration_v10(
        x, x[:, :, 3:4], starts=(0, 0, 3), ends=(20, 10, 4), add_noop_to_input_attrs=["ends"]
    )
    _test_slice_iteration_v10(
        x, x[:, 1:1000], starts=(1,), ends=(1000,), axes=(1,), add_noop_to_input_attrs=["axes"]
    )
    _test_slice_iteration_v10(
        x,
        x[:, 0:-1],
        starts=(0,),
        ends=(-1,),
        axes=(1,),
        add_noop_to_input_attrs=["starts", "ends"],
    )
    _test_slice_iteration_v10(
        x,
        x[0:3, 0:10],
        starts=(0, 0),
        ends=(3, 10),
        axes=(0, 1),
        add_noop_to_input_attrs=["ends", "axes"],
    )
    _test_slice_iteration_v10(
        x,
        x[:, :, 3:4],
        starts=(0, 0, 3),
        ends=(20, 10, 4),
        add_noop_to_input_attrs=["starts", "axes"],
    )
    _test_slice_iteration_v10(
        x,
        x[:, 1:1000],
        starts=(1,),
        ends=(1000,),
        axes=(1,),
        add_noop_to_input_attrs=["starts", "ends", "axes"],
    )
    x = np.random.randn(1, 1, 1, 128).astype(np.float32)
    _test_slice_iteration_v10(
        x, x, starts=(0, 0), ends=(9223372036854775807, 9223372036854775807), axes=(0, 3)
    )

    x = np.random.randn(4, 4).astype(np.float32)
    _test_slice_iteration_v10(
        x, x[:, 1::2], starts=(1,), ends=(9223372036854775807,), axes=(1,), steps=(2,)
    )
    _test_slice_iteration_v10(
        x,
        x[0::1, 1::2],
        starts=(0, 1),
        ends=(4, 4),
        axes=(0, 1),
        steps=(1, 2),
    )


def _test_onnx_op_elementwise(
    target, dev, inshape, outfunc, npargs, dtype, opname, kwargs, opset=None, verify=True
):
    indata = np.random.uniform(-1, 1, size=inshape).astype(dtype)
    outdata = outfunc(indata, **npargs)

    y = helper.make_node(opname, ["in"], ["out"], **kwargs)

    ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", ONNX_DTYPE, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")
    if verify:
        verify_with_ort_with_inputs(
            model, [indata], [outdata.shape], opset=opset, dtype=dtype, target=target, dev=dev
        )
    else:
        get_tinygrad_output(
            model,
            [indata],
            target,
            dev,
            [outdata.shape],
            dtype,
            opset=opset,
            opt_level=3,
        )


@tvm.testing.parametrize_targets
def test_floor(target, dev):
    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), np.floor, {}, "float32", "Floor", {})


@tvm.testing.parametrize_targets
def test_ceil(target, dev):
    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), np.ceil, {}, "float32", "Ceil", {})


@tvm.testing.parametrize_targets
def test_clip(target, dev):
    """test_clip"""
    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -1.0, "a_max": 1.0},
        "float32",
        "Clip",
        {"min": -1.0, "max": 1.0},
        opset=6,
    )

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -np.inf, "a_max": 1.0},
        "float32",
        "Clip",
        {"max": 1.0},
        opset=6,
    )

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -1.0, "a_max": np.inf},
        "float32",
        "Clip",
        {"min": -1.0},
        opset=6,
    )


@tvm.testing.parametrize_targets
def test_clip_min_max_as_inputs(target, dev):
    """test_clip_min_max_as_inputs"""
    input_shape = (2, 4, 5, 6)
    nodes = [
        make_constant_node("min", onnx.TensorProto.FLOAT, (), [0.0]),
        make_constant_node("max", onnx.TensorProto.FLOAT, (), [6.0]),
    ]
    input_names = ["in", "min", "max"]
    nodes.append(helper.make_node("Clip", inputs=input_names, outputs=["out"]))
    graph = helper.make_graph(
        nodes,
        "clip_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(input_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_shape))],
    )
    model = helper.make_model(graph, producer_name="clip_test")

    verify_with_ort(model, [input_shape], out_shape=[input_shape], target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_round(target, dev):
    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), np.round, {}, "float32", "Round", {})
    _test_onnx_op_elementwise(
        target, dev, (2, 4, 5, 6), np.round, {}, "float64", "Round", {}, verify=False
    )  # TODO: enable verification once ORT supports float64


def _test_finite_ops(target, dev, inshape, outfunc, npargs, dtype, opname, kwargs):
    indata = np.random.choice(a=[np.nan, np.inf, -np.inf, 0.5, 1.0, 0], size=inshape).astype(dtype)

    outdata = outfunc(indata, **npargs)
    y = helper.make_node(opname, ["in"], ["out"], **kwargs)

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")
    verify_with_ort_with_inputs(
        model, [indata], [outdata.shape], dtype=dtype, target=target, dev=dev
    )


@tvm.testing.parametrize_targets
def test_isinf(target, dev):
    _test_finite_ops(target, dev, (2, 4, 5, 6), np.isinf, {}, "float32", "IsInf", {})


@tvm.testing.parametrize_targets
def test_isnan(target, dev):
    """test_isnan"""
    _test_finite_ops(target, dev, (2, 4, 5, 6), np.isnan, {}, "float32", "IsNaN", {})


@tvm.testing.parametrize_targets
def test_gather_nd(target, dev):
    """test_gather_nd"""

    def verify_gather_nd(in_shape, indices, out_shape, dtype="float32", batch_dims=0, opset=11):
        x = np.random.uniform(size=in_shape).astype(dtype)
        indices = np.array(indices, dtype="int64")

        y = helper.make_node("GatherND", ["in", "indices"], ["out"])

        if opset >= 12:
            batch_dims_attr = helper.make_attribute("batch_dims", batch_dims)
            y.attribute.append(batch_dims_attr)

        graph = helper.make_graph(
            [y],
            "gather_test",
            inputs=[
                helper.make_tensor_value_info(
                    "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(in_shape)
                ),
                helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(out_shape)
                )
            ],
        )
        model = helper.make_model(graph, producer_name="gather_test")
        verify_with_ort_with_inputs(
            model, [x, indices], [out_shape], opset=opset, target=target, dev=dev
        )

    verify_gather_nd([2, 2], [[0, 0], [1, 1]], [2], "int32")
    verify_gather_nd([2, 2], [[1], [0]], [2, 2])
    verify_gather_nd([2, 2, 2], [[0, 1], [1, 0]], [2, 2])
    verify_gather_nd([2, 2, 2], [[[0, 1]], [[1, 0]]], [2, 1, 2])

    if is_version_greater_than("1.6.0"):
        verify_gather_nd([2, 2, 2], [[1], [0]], [2, 2], batch_dims=1, opset=12)
        verify_gather_nd(
            (3, 2, 2, 3, 4),
            np.random.randint(low=0, high=2, size=(3, 2, 3), dtype="int64"),
            (3, 2),
            batch_dims=2,
            opset=12,
        )


@tvm.testing.parametrize_targets
def test_onehot(target, dev):
    """test_onehot"""
    indices_shape = [10]
    indices_array = np.random.randint(low=0, high=9, size=indices_shape, dtype="int32")
    depth = 10
    values = np.asarray([0, 1]).astype("int32")
    out_np = np.eye(depth)[indices_array.reshape(-1)]

    onehot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["out"])

    graph = helper.make_graph(
        [onehot_node],
        "onehot_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT32, indices_shape),
            helper.make_tensor_value_info("depth", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("values", TensorProto.INT32, values.shape),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT32, out_np.shape)],
    )

    model = helper.make_model(graph, producer_name="onehot_test")

    # TODO(jwfromm): Replace test against np with test against onnxrt once we update versions.
    tvm_out = get_tinygrad_output_with_vm(
        model, [indices_array, np.array([depth]).astype("int32"), values], target, dev
    )
    tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_gemm(target, dev):
    """test_gemm"""

    def verify_gemm(a_shape, b_shape, c_shape=None, freeze_params=False, dtype="float32"):
        out_shape = [a_shape[0], b_shape[1]]
        a_array = np.random.uniform(size=a_shape).astype(dtype)
        b_array = np.random.uniform(size=b_shape).astype(dtype)
        input_names = ["a", "b"]
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        input_nodes = [
            helper.make_tensor_value_info("a", ONNX_DTYPE, list(a_shape)),
            helper.make_tensor_value_info("b", ONNX_DTYPE, list(b_shape)),
        ]
        input_values = [a_array, b_array]
        if c_shape is not None:
            c_array = np.random.uniform(size=c_shape).astype(dtype)
            input_names.append("c")
            input_nodes.append(helper.make_tensor_value_info("c", ONNX_DTYPE, list(c_shape)))
            input_values.append(c_array)

        gemm_node = helper.make_node("Gemm", input_names, ["out"])

        graph = helper.make_graph(
            [gemm_node],
            "gemm_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="gemm_test")
        atol = 1e-5
        rtol = 1e-5
        if dtype == "float16":
            atol = 1e-3
            rtol = 1e-3
        verify_with_ort_with_inputs(
            model,
            input_values,
            freeze_params=freeze_params,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            target=target,
            dev=dev,
        )

    verify_gemm(a_shape=(4, 3), b_shape=(3, 4))
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,))
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,), freeze_params=True)
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,), freeze_params=True, dtype="float16")


@tvm.testing.parametrize_targets
def test_matmul(target, dev):
    """test_matmul"""

    def test_one_matmul(a_shape, b_shape):
        out_shape = np.matmul(np.zeros(a_shape), np.zeros(b_shape)).shape

        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")

        mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

        graph = helper.make_graph(
            [mul_node],
            "matmul_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="matmul_test")
        verify_with_ort_with_inputs(model, [a_array, b_array], target=target, dev=dev)

    test_one_matmul((4, 3), (3, 4))
    test_one_matmul((3,), (3, 1))
    test_one_matmul((1, 3), (3,))
    test_one_matmul((3,), (3,))


@tvm.testing.parametrize_targets
def test_batch_matmul(target, dev):
    """test_batch_matmul"""

    def verify_batch_matmul(a_shape, b_shape, out_shape, convert_config=None):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")

        mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

        graph = helper.make_graph(
            [mul_node],
            "matmul_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name="matmul_test")
        verify_with_ort_with_inputs(
            model,
            [a_array, b_array],
            use_vm=True,
            target=target,
            dev=dev,
            convert_config=convert_config,
        )

    verify_batch_matmul((2, 3, 4, 3), (2, 3, 3, 4), (2, 3, 4, 4))
    verify_batch_matmul((2, 4, 3), (3, 4), (2, 4, 4))
    verify_batch_matmul((2, 3, 4, 3), (3, 4), (2, 3, 4, 4))
    # Test implicit broadcasting.
    verify_batch_matmul((5,), (5, 5, 4), (5, 4))
    verify_batch_matmul((5, 4, 5), (5,), (5, 4))
    verify_batch_matmul((4, 3), (2, 3, 4), (2, 4, 4))
    verify_batch_matmul((2, 4, 3), (1, 3, 4), (2, 4, 4))
    verify_batch_matmul((1, 4, 3), (2, 3, 4), (2, 4, 4))
    verify_batch_matmul((4, 32, 16), (16, 32), (4, 32, 32))
    verify_batch_matmul((4, 32, 16, 32), (32, 16), (4, 32, 16, 16))
    verify_batch_matmul((4, 32, 16, 32), (1, 32, 32, 16), (4, 32, 16, 16))
    verify_batch_matmul((4, 1, 16, 32), (1, 32, 32, 16), (4, 32, 16, 16))
    # Test transb=False
    verify_batch_matmul(
        (2, 3, 4, 3),
        (2, 3, 3, 4),
        (2, 3, 4, 4),
        convert_config={"use_nt_batch_matmul": False},
    )


@tvm.testing.parametrize_targets
def test_use_nt_batch_matmul(target, dev):
    """test_use_nt_batch_matmul"""
    a_shape = (2, 3, 4)
    b_shape = (2, 4, 3)
    out_shape = [2, 3, 3]
    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")

    for use_nt_batch_matmul in [True, False]:
        mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

        graph = helper.make_graph(
            [mul_node],
            "matmul_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="matmul_test")
        _, shape_dict = get_input_data_shape_dict(model, [a_array, b_array])

        mod, _ = relay.frontend.from_onnx(
            model, shape_dict, convert_config={"use_nt_batch_matmul": use_nt_batch_matmul}
        )
        has_transpose_op = "transpose" in str(mod)
        # use_nt_batch_matmul implies, TVM converts qualified onnx `matmul`
        # to `transpose(weight) + nn.batch_matmul_NT`, otherwise to `nn.batch_matmul`
        assert has_transpose_op == use_nt_batch_matmul


@tvm.testing.parametrize_targets
def test_matmulinteger16(target, dev):
    """test_matmulinteger16"""

    def verify_matmulinteger16(a_shape, b_shape, out_shape):
        a_dtype = "int16"
        b_dtype = "int16"
        low = np.iinfo(np.int16).min
        high = np.iinfo(np.int16).max

        a_proto = TensorProto.INT16
        b_proto = TensorProto.INT16
        out_proto = TensorProto.INT32
        a_array = np.random.randint(low, high, size=a_shape).astype(a_dtype)
        b_array = np.random.randint(low, high, size=b_shape).astype(b_dtype)

        mul_node = helper.make_node("MatMulInteger16", ["a", "b"], ["out"], domain="com.microsoft")

        graph = helper.make_graph(
            [mul_node],
            "matmuli16_test",
            inputs=[
                helper.make_tensor_value_info("a", a_proto, list(a_shape)),
                helper.make_tensor_value_info("b", b_proto, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", out_proto, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="matmuli16_test")
        verify_with_ort_with_inputs(model, [a_array, b_array], target=target, dev=dev)

    # 2D computation to verify matmul op
    verify_matmulinteger16((4, 3), (3, 4), (4, 4))
    verify_matmulinteger16((5, 7), (7, 8), (5, 8))
    # Verify 3D matmul using batch_matmul op
    verify_matmulinteger16((2, 4, 3), (1, 3, 4), (2, 4, 4))
    verify_matmulinteger16((1, 4, 3), (2, 3, 4), (2, 4, 4))
    # Test implicit broadcasting
    verify_matmulinteger16((2, 3, 5, 3), (2, 3, 3, 5), (2, 3, 5, 5))
    verify_matmulinteger16((2, 7, 3), (3, 7), (2, 7, 7))
    verify_matmulinteger16((2, 3, 4, 3), (3, 4), (2, 3, 4, 4))


def verify_simple_dynamic_model(a_shape, b_shape, target, dev):
    """verify_simple_dynamic_model"""

    def verify_model(model, a_shape, b_shape):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")
        # matmul
        out_np = np.matmul(a_array, b_array)
        # relu
        out_np[out_np < 0] = 0

        tvm_out = model(a_array, b_array).numpy()
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])
    relu_node = helper.make_node("Relu", ["out"], ["relu"])

    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")
    # matmul
    out_np = np.matmul(a_array, b_array)

    graph = helper.make_graph(
        [mul_node, relu_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ],
        outputs=[helper.make_tensor_value_info("relu", TensorProto.FLOAT, list(out_np.shape))],
    )

    model = helper.make_model(graph, producer_name="matmul_test")

    a_anys = [relay.Any()] * len(a_shape)
    b_anys = [relay.Any()] * len(b_shape)

    mod, _ = relay.frontend.from_onnx(model, {"a": a_anys, "b": b_anys})
    model = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()
    verify_model(model, a_shape, b_shape)
    verify_model(model, [a * 2 for a in a_shape], [b * 2 for b in b_shape])
    verify_model(model, [a * 3 for a in a_shape], [b * 3 for b in b_shape])


# TODO(mbrookhart, electriclilies): Add CUDA as a target once batch matmul is fixed
@tvm.testing.parametrize_targets("llvm")
def test_batch_matmul_dynamic_model(target, dev):
    verify_simple_dynamic_model((2, 3, 4, 3), (2, 3, 3, 4), target, dev)
    verify_simple_dynamic_model((2, 4, 3), (3, 4), target, dev)
    verify_simple_dynamic_model((2, 3, 4, 3), (3, 4), target, dev)


@tvm.testing.parametrize_targets
def test_lrn(target, dev):
    """test_lrn"""

    def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None):
        in_array = np.random.uniform(size=shape).astype(dtype)

        if alpha is None and beta is None and bias is None:
            alpha = 0.0001
            beta = 0.75
            bias = 1.0
            node = onnx.helper.make_node("LRN", inputs=["in"], outputs=["out"], size=nsize)
        else:
            node = onnx.helper.make_node(
                "LRN", inputs=["in"], outputs=["out"], alpha=alpha, beta=beta, bias=bias, size=nsize
            )

        graph = helper.make_graph(
            [node],
            "lrn_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))],
        )
        model = helper.make_model(graph, producer_name="lrn_test")
        verify_with_ort_with_inputs(model, [in_array], target=target, dev=dev)

    verify_lrn((5, 5, 5, 5), 3, "float32")
    verify_lrn((5, 5, 5, 5), 3, "float32", alpha=0.0002, beta=0.5, bias=2.0)


@tvm.testing.parametrize_targets
def test_instance_norm(target, dev):
    """test_instance_norm"""

    def verify_instance_norm(shape, axis=1):
        x = np.random.randn(*shape).astype(np.float32)
        gamma = np.random.randn(shape[1]).astype(np.float32)
        beta = np.random.randn(shape[1]).astype(np.float32)
        epsilon = 1e-5

        node = onnx.helper.make_node(
            "InstanceNormalization",
            inputs=["x", "gamma", "beta"],
            outputs=["y"],
            epsilon=epsilon,
        )
        graph = helper.make_graph(
            [node],
            "instance_norm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape)),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, (shape[1],)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, (shape[1],)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))],
        )
        model = helper.make_model(graph, producer_name="instance_norm_test")
        verify_with_ort_with_inputs(
            model, [x, gamma, beta], out_shape=[shape], target=target, dev=dev
        )

    verify_instance_norm((2, 3, 4, 5))
    verify_instance_norm((32, 64, 80, 64))
    verify_instance_norm((8, 6, 5))
    verify_instance_norm((8, 7, 6, 5, 4))


@tvm.testing.parametrize_targets
def test_upsample_nearest(target, dev):
    """test_upsample_nearest"""
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample_nearest_default(target, dev):
    """test_upsample_nearest_default"""
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample3d_nearest(target, dev):
    """test_upsample3d_nearest"""
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node(
        "Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0, 2.0]
    )

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    # Upsample is deprecated after opset 9
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample_bilinear(target, dev):
    """test_upsample_bilinear"""
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="linear", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_bilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_bilinear_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample3d_trilinear(target, dev):
    """test_upsample3d_trilinear"""
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in", "scales"], ["out"], mode="linear")
    scales = [1.0, 1.0, 2.0, 2.0, 2.0]
    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.resize3d_python(
        in_array,
        (scale, scale, scale),
        "NCDHW",
        "linear",
        coordinate_transformation_mode="asymmetric",
    )

    ref_array = np.array(scales)
    ref_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scales"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(float),
        ),
    )

    graph = helper.make_graph(
        [ref_node, y],
        "upsample_trilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_trilinear_test")
    # TODO(jwfromm): Trilinear upsampling not supported in 1.0.0 onnxruntime.
    # Replace topi comparison with verify_with_ort once we update.
    tvm_out = get_tinygrad_output(model, in_array, target, dev, out_shape, "float32")
    tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)


# TODO: Fix softmax with dynamic input on cuda and enable this test
@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_softmax(target, dev):
    """test_softmax"""

    def verify_softmax(inshape, axis, opset=None, dynamic=False):
        opname = "Softmax"
        outshape = inshape
        node_list = []
        input_node_list = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(inshape))]
        output_node_list = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outshape))]
        input_list = [np.random.uniform(size=inshape).astype(np.float32)]
        softmax_inputs = ["in"]

        if dynamic:
            input_node_list.append(
                helper.make_tensor_value_info("shape", TensorProto.INT64, [len(inshape)])
            )
            input_list.append(np.asarray(inshape))
            reshape_node = helper.make_node("Reshape", ["in", "shape"], ["dynamic_in"])
            softmax_inputs[0] = "dynamic_in"
            node_list += [reshape_node]

        y = helper.make_node(opname, softmax_inputs, ["out"])
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            y.attribute.append(axis_attr)
        node_list.append(y)

        graph = helper.make_graph(
            node_list,
            opname + "_test",
            inputs=input_node_list,
            outputs=output_node_list,
        )

        model = helper.make_model(graph, producer_name=opname + "_test")
        verify_with_ort_with_inputs(
            model, input_list, use_vm=True, opset=opset, target=target, dev=dev
        )

    verify_softmax((1, 10), None)
    verify_softmax((1, 10), 1)
    verify_softmax((1, 2, 3, 10), 0)
    verify_softmax((1, 2, 3, 10), 2)
    verify_softmax((1, 2, 3, 4, 10), 3)
    verify_softmax((1, 2, 3, 4, 10), 4)
    verify_softmax((1, 10), -1, dynamic=True)
    verify_softmax((1, 2, 3, 10), -1, dynamic=True)
    verify_softmax((1, 10), -1, opset=8, dynamic=True)
    verify_softmax((1, 2, 3, 10), -1, opset=8, dynamic=True)


@tvm.testing.parametrize_targets
def test_forward_min(target, dev):
    """test_forward_min"""

    def verify_min(input_dim):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)
        a_np2 = np.random.uniform(size=input_dim).astype(dtype)
        a_np3 = np.random.uniform(size=input_dim).astype(dtype)

        min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

        graph = helper.make_graph(
            [min_node],
            "Min_test",
            inputs=[
                helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="Min_test")
        verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3], target=target, dev=dev)

    verify_min((1, 3, 20, 20))
    verify_min((20, 20))


@tvm.testing.parametrize_targets
def test_forward_max(target, dev):
    """test_forward_max"""

    def verify_max(input_dim):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)
        a_np2 = np.random.uniform(size=input_dim).astype(dtype)
        a_np3 = np.random.uniform(size=input_dim).astype(dtype)

        max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

        graph = helper.make_graph(
            [max_node],
            "Max_test",
            inputs=[
                helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="Max_test")
        verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3], target=target, dev=dev)

    verify_max((1, 3, 20, 20))
    verify_max((20, 20))


@tvm.testing.parametrize_targets
def test_forward_mean(target, dev):
    """test_forward_mean"""

    def verify_mean(input_dim):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)
        a_np2 = np.random.uniform(size=input_dim).astype(dtype)
        a_np3 = np.random.uniform(size=input_dim).astype(dtype)

        mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

        graph = helper.make_graph(
            [mean_node],
            "Mean_test",
            inputs=[
                helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="Mean_test")
        verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3], target=target, dev=dev)

    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))


@tvm.testing.parametrize_targets
def test_forward_hardsigmoid(target, dev):
    """test_forward_hardsigmoid"""

    def verify_hardsigmoid(input_dim, alpha, beta):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)

        hardsigmoid_node = helper.make_node(
            "HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta
        )

        graph = helper.make_graph(
            [hardsigmoid_node],
            "HardSigmoid_test",
            inputs=[helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="HardSigmoid_test")
        verify_with_ort_with_inputs(model, [a_np1], target=target, dev=dev)

    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)


# TODO (mbrookhart, electriclilies) Fix argmin on GPU and enable this test
@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_forward_arg_min_max(target, dev):
    """test_forward_arg_min_max"""

    def verify_argreduce(input_dim, op_name, axis=None, keepdims=None):
        a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
        out_shape = list(a_np1.shape)
        def_axis = axis if axis is not None else 0
        if keepdims == 1 or keepdims is None:
            out_shape[def_axis] = 1
        else:
            out_shape.pop(def_axis)

        node = onnx.helper.make_node(op_name, inputs=["a_np1"], outputs=["out"])

        if keepdims is not None:
            keepdims_attr = helper.make_attribute("keepdims", keepdims)
            node.attribute.append(keepdims_attr)
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            node.attribute.append(axis_attr)

        graph = helper.make_graph(
            [node],
            "argreduce_test",
            inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, list(a_np1.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="argreduce_test")
        verify_with_ort_with_inputs(model, [a_np1], target=target, dev=dev)

    # Verify argmin and argmax
    verify_argreduce([3, 4, 4], "ArgMin")
    verify_argreduce([3, 4, 4], "ArgMax")
    verify_argreduce([3, 4, 4], "ArgMin", axis=1)
    verify_argreduce([3, 4, 4], "ArgMax", axis=0)
    verify_argreduce([3, 4, 4], "ArgMin", keepdims=0)
    verify_argreduce([3, 4, 4], "ArgMax", keepdims=1)
    for axis in [None, 0, 1, 2]:
        for keepdims in [None, True, False]:
            verify_argreduce([3, 4, 4], "ArgMin", axis, keepdims)
            verify_argreduce([3, 4, 4], "ArgMax", axis, keepdims)


@tvm.testing.parametrize_targets
def test_constantofshape(target, dev):
    """test_constantofshape"""

    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], (1,), (value,)
            ),
        )

        inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

        graph = helper.make_graph(
            [fill_node],
            "fill_test",
            inputs,
            outputs=[
                helper.make_tensor_value_info(
                    "output", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        input_np = np.array(input_dim).astype("int64")
        verify_with_ort_with_inputs(model, [input_np], use_vm=True, target=target, dev=dev)

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


@tvm.testing.parametrize_targets
def test_pad(target, dev):
    """test_pad"""

    def verify_pad(indata, pads, mode="constant", value=0.0):
        indata = np.array(indata).astype(np.float32)
        #  numpy expect result
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(indata, pad_width=np_pads, mode=mode)
            node = helper.make_node(
                "Pad",
                inputs=["input"],
                outputs=["output"],
                mode=mode,
                pads=pads,
            )
        else:
            outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
            node = helper.make_node(
                "Pad", inputs=["input"], outputs=["output"], mode="constant", pads=pads, value=value
            )
        graph = helper.make_graph(
            [node],
            "pad_test",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
            ],
        )
        model = helper.make_model(graph, producer_name="pad_test")
        verify_with_ort_with_inputs(
            model, [indata], [outdata.shape], dtype="float32", opset=2, target=target, dev=dev
        )

    def verify_pad_v11(indata, pads, mode="constant", value=0.0):
        indata = np.array(indata).astype(np.float32)
        #  numpy expect result
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            inputs = [indata]
            outdata = np.pad(indata, pad_width=np_pads, mode=mode)
            node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                    helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
                ],
                initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            inputs = [indata]
            outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
            node = helper.make_node(
                "Pad",
                inputs=["input", "pads", "constant_value"],
                outputs=["output"],
                mode="constant",
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                    helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
                    helper.make_tensor_value_info("constant_value", TensorProto.FLOAT, (1,)),
                ],
                initializer=[
                    helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                    helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        verify_with_ort_with_inputs(model, inputs, opset=11, use_vm=True, target=target, dev=dev)

    verify_pad(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)
    verify_pad(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], "constant", 0.0)
    verify_pad(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], "constant", 5.0)
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "edge")
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "reflect")

    verify_pad_v11(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)
    verify_pad_v11(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], "constant", 0.0)
    verify_pad_v11(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], "constant", 5.0)
    verify_pad_v11(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "edge")
    verify_pad_v11(
        np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "reflect"
    )


@tvm.testing.parametrize_targets
def test_all_reduce_funcs(target, dev):
    """test_all_reduce_funcs"""

    def verify_reduce_func(func, data, axis, keepdims):
        inshape = data.shape
        outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

        if axis:
            node = onnx.helper.make_node(
                func, inputs=["x"], outputs=["y"], axes=axis, keepdims=keepdims
            )
        else:
            node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], keepdims=keepdims)

        graph = helper.make_graph(
            [node],
            "reduce_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
        )

        model = helper.make_model(graph, producer_name="reduce_test")

        verify_with_ort_with_inputs(
            model,
            [data],
            [outshape],
            opset=11,
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
        )

    funcs = [
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceL1",
        "ReduceL2",
    ]

    for func in funcs:
        verify_reduce_func(func, np.array(1.0).astype(np.float32), axis=None, keepdims=False)

        for keepdims in [True, False]:
            verify_reduce_func(
                func, np.random.randn(3, 2, 2).astype(np.float32), axis=None, keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 2, 3).astype(np.float32), axis=None, keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3).astype(np.float32), axis=(1,), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1, 2), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1,), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(1, 3, 4, 1).astype(np.float32), axis=(1,), keepdims=keepdims
            )


@tvm.testing.parametrize_targets
def test_split(target, dev):
    """test_split"""

    def verify_split(indata, outdatas, split, axis=0, pass_split=True, opset=11):
        indata = np.array(indata).astype(np.float32)
        outdatas = [np.array(o).astype(np.float32) for o in outdatas]
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))]
        input_names = ["input"]
        initializer = []

        if split:
            split_index = range(len(split))
        else:
            split_index = range(len(outdatas))

        if pass_split:
            if opset >= 13:
                input_names.append("split")
                np_split = np.array(split).astype(np.int64)
                inputs.append(
                    helper.make_tensor_value_info("split", TensorProto.INT64, list(np_split.shape))
                )
                # TODO(mbrookhart): Support dynamic split, edit this test case to remove split from
                # the initializer and add it back to the input data
                indata = [indata]  # , np_split]
                initializer.append(
                    helper.make_tensor("split", TensorProto.INT64, list(np_split.shape), np_split)
                )
        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        graph = helper.make_graph(
            [node],
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}", TensorProto.FLOAT, list(outdatas[i].shape)
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        verify_with_ort_with_inputs(
            model,
            indata,
            out_shape=list(range(len(split_index))),
            opset=opset,
            target=target,
            dev=dev,
            use_vm=True,
            freeze_params=(opset >= 13),
        )

    # 1D
    verify_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2, 2, 2], 0)
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2, 2, 2], 0, False
    )
    verify_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], [2, 1, 3], 0)
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], [2, 1, 3], 0, opset=13
    )
    # 2D
    verify_split(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]],
        [[[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [9.0, 10.0]]],
        [2, 2],
        1,
    )
    verify_split(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]],
        [[[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [9.0, 10.0]]],
        [2, 2],
        1,
        opset=13,
    )
    # Split evenly (unstack)
    verify_split([1, 2, 3], [[1], [2], [3]], False, 0, False)
    # Split a single value to a single value
    verify_split([1], [[1]], [1], pass_split=True)
    # Test that the default case modifies nothing when split list has length one
    verify_split([[1.0, 2.0]], [[1.0, 2.0]], [2], 1)
    verify_split([[1.0, 2.0]], [[1.0, 2.0]], [1], 0)


@tvm.testing.parametrize_targets
def test_binary_ops(target, dev):
    """test_binary_ops"""
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_type="float32"):
        out = helper.make_node(op, ["in1", "in2"], ["out"])
        graph = helper.make_graph(
            [out],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.FLOAT, x.shape),
                helper.make_tensor_value_info("in2", TensorProto.FLOAT, y.shape),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_type)], list(out_shape)
                )
            ],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_inputs(model, [x, y], target=target, dev=dev)

    x = np.random.uniform(size=in_shape).astype(dtype)
    y = np.random.uniform(size=in_shape).astype(dtype)
    z_array = np.random.uniform(size=(3,)).astype(dtype)
    verify_binary_ops("Add", x, y)
    verify_binary_ops("Add", x, z_array)
    verify_binary_ops("Sub", x, y)
    verify_binary_ops("Sub", x, z_array)
    verify_binary_ops("Mul", x, y)
    verify_binary_ops("Mul", x, z_array)
    verify_binary_ops("Div", x, y)
    verify_binary_ops("Div", x, z_array)
    verify_binary_ops("Sum", x, y)
    verify_binary_ops("Sum", x, z_array)
    verify_binary_ops("Greater", x, y, "bool")
    verify_binary_ops("Greater", x, z_array, "bool")
    verify_binary_ops("GreaterOrEqual", x, y, "bool")
    verify_binary_ops("GreaterOrEqual", x, z_array, "bool")
    verify_binary_ops("Less", x, y, "bool")
    verify_binary_ops("Less", x, z_array, "bool")
    verify_binary_ops("LessOrEqual", x, y, "bool")
    verify_binary_ops("LessOrEqual", x, z_array, "bool")
    verify_binary_ops("Equal", x, y, "bool")
    verify_binary_ops("Equal", x, z_array, "bool")


@tvm.testing.parametrize_targets
def test_unary_ops(target, dev):
    """test_unary_ops"""
    in_shape = (1, 2, 3, 3)
    _ = "float32"
    out_shape = in_shape

    def verify_unary_ops(op, x, rtol=1e-5, atol=1e-5, dtype="float32"):
        x = x.astype(dtype)
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        out = helper.make_node(op, ["in1"], ["out"])
        graph = helper.make_graph(
            [out],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", ONNX_DTYPE, list(in_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_inputs(model, [x], rtol=rtol, atol=atol, target=target, dev=dev)

    x = np.random.uniform(size=in_shape)
    verify_unary_ops("Neg", x)
    verify_unary_ops("Abs", x)
    verify_unary_ops("Reciprocal", x)
    verify_unary_ops("Reciprocal", x, dtype="float16")
    verify_unary_ops("Sqrt", x)
    verify_unary_ops("Relu", x)
    verify_unary_ops("Exp", x)
    verify_unary_ops("Log", x)
    verify_unary_ops("Log", x)
    verify_unary_ops("Acos", x)
    verify_unary_ops("Acosh", x)
    verify_unary_ops("Asin", x)
    verify_unary_ops("Asinh", x)
    verify_unary_ops("Atan", x)
    verify_unary_ops("Atanh", x)
    verify_unary_ops("Cos", x)
    verify_unary_ops("Cosh", x)
    verify_unary_ops("Sin", x)
    verify_unary_ops("Sinh", x)
    verify_unary_ops("Tan", x)
    verify_unary_ops("Tanh", x)
    verify_unary_ops("Sigmoid", x)
    verify_unary_ops("Softsign", x)


@tvm.testing.parametrize_targets
def test_leaky_relu(target, dev):
    def leaky_relu_x(x, alpha):
        return np.where(x >= 0, x, x * alpha)

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        leaky_relu_x,
        {"alpha": 0.25},
        "float32",
        "LeakyRelu",
        {"alpha": 0.25},
    )


@tvm.testing.parametrize_targets
def test_elu(target, dev):
    def elu_x(x, alpha):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    _test_onnx_op_elementwise(
        target, dev, (2, 4, 5, 6), elu_x, {"alpha": 0.25}, "float32", "Elu", {"alpha": 0.25}
    )


@tvm.testing.parametrize_targets
def test_selu(target, dev):
    def selu_x(x, alpha, gamma):
        return gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        selu_x,
        {"alpha": 0.25, "gamma": 0.3},
        "float32",
        "Selu",
        {"alpha": 0.25, "gamma": 0.3},
    )


@pytest.mark.skip("Currently ONNX Runtime in CI does not support domain version of 18")
@tvm.testing.parametrize_targets
def test_mish(target, dev):
    def mish_x(x):
        return x * np.tanh(np.log1p(np.exp(x)))

    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), mish_x, {}, "float64", "Mish", {})


@tvm.testing.parametrize_targets
def test_prelu(target, dev):
    """test_prelu"""

    def verify_prelu(x_shape, a_shape):
        node = helper.make_node("PRelu", inputs=["X", "slope"], outputs=["Y"])

        graph = helper.make_graph(
            [node],
            "prelu_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("slope", TensorProto.FLOAT, list(a_shape)),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x_shape))],
        )

        model = helper.make_model(graph, producer_name="prelu_test")

        verify_with_ort(
            model,
            [x_shape, a_shape],
            out_shape=[list(x_shape)],
            use_vm=True,
            target=target,
            dev=dev,
        )

    verify_prelu([3, 4, 5, 6], [1, 4, 1, 1])
    verify_prelu([1, 8, 5, 6], [1, 8, 1, 1])
    verify_prelu([2, 12, 16, 16], [1, 12, 1, 1])
    verify_prelu([2, 12, 16, 16], [1])  # Test alpha broadcasting.
    verify_prelu([3, 1], [3, 1])  # Test non NCHW workload.


@tvm.testing.parametrize_targets
def test_thresholded_relu(target, dev):
    def thresholded_relu_x(x, alpha):
        out_np = np.clip(x, alpha, np.inf)
        out_np[out_np == alpha] = 0
        return out_np

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        thresholded_relu_x,
        {"alpha": 0.25},
        "float32",
        "ThresholdedRelu",
        {"alpha": 0.25},
    )


@tvm.testing.parametrize_targets
def test_logsoftmax(target, dev):
    _test_onnx_op_elementwise(
        target,
        dev,
        (1, 4),
        tvm.topi.testing.log_softmax_python,
        {},
        "float32",
        "LogSoftmax",
        {"axis": 1},
    )


def check_torch_conversion(model, input_size, target, dev):
    dummy_input = torch.randn(*input_size)
    file_name = f"{model.__name__}.onnx"
    # Set verbose=True for more output
    torch.onnx.export(model(), dummy_input, file_name, export_params=True, verbose=False)
    onnx_model = onnx.load(file_name)
    input_data = np.random.uniform(size=input_size).astype("float32")
    verify_with_ort_with_inputs(
        onnx_model, [input_data], apply_softmax=True, target=target, dev=dev
    )


@tvm.testing.parametrize_targets
def test_resnet(target, dev):
    check_torch_conversion(torchvision.models.resnet18, (1, 3, 224, 224), target, dev)
    # check_torch_conversion(torchvision.models.resnet101, (1,3,224,224))


# def test_alexnet():
# Torch's ONNX export does not support the adaptive pooling used by AlexNet?
# check_torch_conversion(torchvision.models.alexnet, (1,3,224,224))

# Torch's ONNX export does not support the adaptive pooling used by vgg16?
# def test_vgg16():
#     check_torch_conversion(torchvision.models.vgg16, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_squeezenet():
#     # Torch's ONNX export does not support the max pooling used by Squezenet
#     check_torch_conversion(torchvision.models.squeezenet1_0, (1,3,224,224))


@tvm.testing.parametrize_targets
def test_densenet(target, dev):
    check_torch_conversion(torchvision.models.densenet161, (1, 3, 224, 224), target, dev)


@tvm.testing.parametrize_targets
def test_inception(target, dev):
    check_torch_conversion(torchvision.models.inception_v3, (1, 3, 224, 224), target, dev)


# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_googlenet():
#     check_torch_conversion(torchvision.models.googlenet, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_shufflenetv2():
#     check_torch_conversion(torchvision.models.shufflenetv2, (1,3,224,224))


@tvm.testing.parametrize_targets
def test_sign(target, dev):
    def sign_x(x):
        return np.sign(x)

    _test_onnx_op_elementwise(target, dev, (3, 4, 5, 6), sign_x, {}, "float32", "Sign", {})


@tvm.testing.parametrize_targets
def test_not(target, dev):
    """test_not"""

    def verify_not(indata, dtype):
        x = indata.astype(dtype)

        node = helper.make_node(
            "Not",
            inputs=["in"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "not_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.BOOL, list(x.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name="not_test")
        verify_with_ort_with_inputs(model, [x], target=target, dev=dev)

    # 2d
    verify_not(indata=(np.random.randn(3, 4) > 0), dtype=bool)
    # 3d
    verify_not(indata=(np.random.randn(3, 4, 5) > 0), dtype=bool)
    # 4d
    verify_not(indata=(np.random.randn(3, 4, 5, 6) > 0), dtype=bool)


@tvm.testing.parametrize_targets
def test_and(target, dev):
    """test_and"""

    def verify_and(indata, dtype):
        x = indata[0].astype(dtype)
        y = indata[1].astype(dtype)
        outdata = np.logical_and(x, y)

        node = helper.make_node(
            "And",
            inputs=["in1", "in2"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "and_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
                helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="and_test")
        verify_with_ort_with_inputs(model, [x, y], [outdata.shape], target=target, dev=dev)

    # 2d
    x = np.random.randn(3, 4) > 0
    y = np.random.randn(3, 4) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 4d
    x = np.random.randn(3, 4, 5, 6) > 0
    y = np.random.randn(3, 4, 5, 6) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(5) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    verify_and(indata=[x, y], dtype=bool)


@tvm.testing.parametrize_targets
def test_tile(target, dev):
    """test_tile"""

    def verify_tile_v6(indata, repeats, outdata):
        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])
        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                helper.make_tensor_value_info("repeats", TensorProto.INT64, list(repeats.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="tile_test")
        verify_with_ort_with_inputs(
            model, [indata, repeats], use_vm=True, opset=6, target=target, dev=dev
        )

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z_array = np.tile(x, repeats)
    verify_tile_v6(x, repeats, z_array)


@tvm.testing.parametrize_targets
def test_erf(target, dev):
    """test_erf"""

    def verify_erf(indata, outdata):
        node = helper.make_node("Erf", inputs=["in"], outputs=["out"])
        graph = helper.make_graph(
            [node],
            "erf_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
        )
        model = helper.make_model(graph, producer_name="erf_test")
        verify_with_ort_with_inputs(model, [indata], [outdata.shape], target=target, dev=dev)

    x = np.random.rand(2, 3, 4, 6).astype(np.float32)
    z_array = scipy.special.erf(x)
    verify_erf(x, z_array)


@tvm.testing.parametrize_targets
def test_where(target, dev):
    """test_where"""

    def verify_where(condition, x, y, dtype, outdata, dynamic=False):
        node_list = []
        where_inputs = ["condition", "x", "y"]
        if dynamic:
            shape_node = helper.make_node("Shape", ["x"], ["shape"])
            reshape_node = helper.make_node("Reshape", ["x", "shape"], ["X"])
            where_inputs[1] = "X"
            node_list += [shape_node, reshape_node]
        node = helper.make_node("Where", inputs=where_inputs, outputs=["out"])
        node_list.append(node)
        graph = helper.make_graph(
            node_list,
            "where_test",
            inputs=[
                helper.make_tensor_value_info("condition", TensorProto.BOOL, list(condition.shape)),
                helper.make_tensor_value_info("x", dtype, list(x.shape)),
                helper.make_tensor_value_info("y", dtype, list(y.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", dtype, list(outdata.shape))],
        )
        model = helper.make_model(graph, producer_name="where_test")
        verify_with_ort_with_inputs(
            model, [condition, x, y], [outdata.shape], use_vm=True, target=target, dev=dev
        )

    condition = np.array([[1, 0], [1, 1]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[9, 8], [7, 6]], dtype=np.int64)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.INT64, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array(1, dtype=np.float32)
    y = np.array([2], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([2], dtype=np.float32)
    y = np.array(1, dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    condition = np.array(1, dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[1], [7]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata, dynamic=True)

    condition = np.random.uniform(size=(3, 1)) < 0.5
    x = np.random.uniform(size=2).astype(np.float32)
    y = np.random.uniform(size=2).astype(np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)


@tvm.testing.parametrize_targets
def test_or(target, dev):
    """test_or"""

    def verify_or(indata, dtype):
        x = indata[0].astype(dtype)
        y = indata[1].astype(dtype)
        outdata = np.logical_or(x, y)

        node = helper.make_node(
            "Or",
            inputs=["in1", "in2"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "or_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
                helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="or_test")
        verify_with_ort_with_inputs(model, [x, y], [outdata.shape], target=target, dev=dev)

    # 2d
    x = np.random.randn(3, 4) > 0
    y = np.random.randn(3, 4) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 4d
    x = np.random.randn(3, 4, 5, 6) > 0
    y = np.random.randn(3, 4, 5, 6) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(5) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    verify_or(indata=[x, y], dtype=bool)


@tvm.testing.parametrize_targets
def test_batch_norm(target, dev):
    """test_batch_norm"""

    def verify_batch_norm(in_shape):
        batchnorm = onnx.helper.make_node(
            "BatchNormalization", inputs=["x", "scale", "B", "mean", "var"], outputs=["Y"]
        )

        graph = helper.make_graph(
            [batchnorm],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")
        # X, scale, b, mean, var
        inshapes = [in_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        verify_with_ort(model, inshapes, out_shape=[in_shape], target=target, dev=dev)

    verify_batch_norm([1, 3, 224, 224])
    verify_batch_norm([1, 3, 24, 24])
    verify_batch_norm([16, 3, 24, 24])
    verify_batch_norm([16, 16, 24, 24])
    verify_batch_norm([16, 16, 10, 10])


@tvm.testing.parametrize_targets
def test_batch_norm_dynamic_subgraph(target, dev):
    """test_batch_norm_dynamic_subgraph"""

    def verify_batch_norm_dynamic_subgraph(in_shape, o_shape):

        batchnorm = onnx.helper.make_node(
            "BatchNormalization", inputs=["x", "scale", "B", "mean", "var"], outputs=["Y"]
        )

        shape_node = helper.make_node("Shape", ["Y"], ["shape"])
        reshape_node = helper.make_node("Reshape", ["in", "shape"], ["out"])
        graph = helper.make_graph(
            [batchnorm, shape_node, reshape_node],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("in", TensorProto.FLOAT, list(o_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")

        # X, inp, scale, b, mean, var
        inshapes = [in_shape, o_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        verify_with_ort(model, inshapes, out_shape=[in_shape], use_vm=True, target=target, dev=dev)

    verify_batch_norm_dynamic_subgraph([16, 16, 10, 10], [160, 160])


@tvm.testing.parametrize_targets
def test_conv(target, dev):
    """test_conv"""

    def verify_conv(
        x_shape,
        w_shape,
        y_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        group=1,
        auto_pad="NOTSET",
        unset_pad=False,
    ):
        if unset_pad:
            node = helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                group=group,
            )
        elif padding is None:
            ## autopadding with unset default attributes
            kwargs = {}
            if not all(list(s == 1 for s in strides)):
                kwargs["strides"] = strides
            if not all(list(d == 1 for d in dilations)):
                kwargs["dilations"] = dilations

            node = helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                # Default values for other attributes:
                auto_pad=auto_pad,
                group=group,
                **kwargs,
            )
        else:
            node = helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                group=group,
                pads=padding,
            )

        graph = helper.make_graph(
            [node],
            "conv_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
        )

        model = helper.make_model(graph, producer_name="conv_test")

        verify_with_ort(
            model,
            [x_shape, w_shape],
            [y_shape],
            use_vm=True,
            target=target,
            dev=dev,
        )

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    for dims in [1, 2, 3]:
        # Convolution with padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(5, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution with asymmetric padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(4, dims),
            repeat(0, dims) + repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution without padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution with autopadding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(5, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with valid autopadding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="VALID",
        )
        # Convolution with unset padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            True,
        )
        # Convolution with non uniform stride
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(2, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with dilation
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(5, dims),
            2 * repeat(2, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(2, dims),
        )

    # TODO(jwfromm): Merge with other tests once group_conv3d is supported.
    for dims in [1, 2, 3]:
        # Group Convolution
        verify_conv(
            (1, 8) + repeat(5, dims),
            (8, 1) + repeat(3, dims),
            (1, 8) + repeat(5, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            group=8,
        )

        verify_conv(
            (1, 12) + repeat(5, dims),
            (30, 4) + repeat(3, dims),
            (1, 30) + repeat(5, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            group=3,
        )


@tvm.testing.parametrize_targets
def test_convtranspose(target, dev):
    """test_convtranspose"""

    def verify_convtranspose_with_output_shape(
        x_shape,
        w_shape,
        output_shape,
        kernel_shape,
        strides,
        dilations,
        auto_pad="SAME_UPPER",
        group=1,
    ):
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            output_shape=output_shape,
            auto_pad=auto_pad,
        )

        if group is not None:
            group_attr = helper.make_attribute("group", group)
            node.attribute.append(group_attr)

        graph = helper.make_graph(
            [node],
            "ConvTranspose_with_output_shape_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1] + list(output_shape))
            ],
        )

        model = helper.make_model(graph, producer_name="convtranspose_output_shape_test")

        verify_with_ort(model, [x_shape, w_shape], use_vm=True, target=target, dev=dev)

    def verify_convtranspose_with_padding(
        x_shape,
        w_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        auto_pad="NOTSET",
        unset_pad=False,
        group=1,
    ):
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
        )
        if not unset_pad:
            if padding is None:
                pad_attr = helper.make_attribute("auto_pad", auto_pad)
            else:
                pad_attr = helper.make_attribute("pads", padding)
            node.attribute.append(pad_attr)

        if group is not None:
            group_attr = helper.make_attribute("group", group)
            node.attribute.append(group_attr)

        graph = helper.make_graph(
            [node],
            "convtranspose_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["?"] * len(x_shape))],
        )

        model = helper.make_model(graph, producer_name="convtranspose_pad_test")

        verify_with_ort(model, [x_shape, w_shape], use_vm=True, target=target, dev=dev)

    def verify_convtranspose(x_shape, w_shape, y_shape, p, group=1):
        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            strides=[3, 2],
            kernel_shape=[3, 3],
            pads=p,
        )

        if group is not None:
            group_attr = helper.make_attribute("group", group)
            node.attribute.append(group_attr)

        graph = helper.make_graph(
            [node],
            "verify_convtranspose_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
        )

        model = helper.make_model(graph, producer_name="convtranspose_test")
        verify_with_ort(model, [x_shape, w_shape], y_shape, opset=11, target=target, dev=dev)

    # Convolution Transpose with padding
    # (1, 1, 3, 3) input tensor
    # (1, 2, 3, 3) tensor for convolution weights
    # (1, 2, 7, 3) output tensor
    # [1, 2, 1, 2] list for pads
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2])
    # Test undefined groups.
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2], group=None)

    if "llvm" in target:
        # GPU does not support groups != 1 for convtranspose, so only test llvm
        # Test depthwise-convolution
        verify_convtranspose((1, 10, 3, 3), (10, 1, 3, 3), (1, 10, 7, 3), [1, 2, 1, 2], group=10)

        # Test grouped-convolution
        verify_convtranspose((1, 10, 3, 3), (10, 1, 3, 3), (1, 5, 7, 3), [1, 2, 1, 2], group=5)

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    # Once onnxruntime update is complete
    for dims in [1, 2, 3]:
        # Convolution with padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution without padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution with unset padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            True,
        )
        # Convolution with autopadding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with valid autopadding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="VALID",
        )
        # Convolution with non uniform stride
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(2, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with default stride
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            None,
            repeat(1, dims),
        )
        # Convolution with dilation
        # TODO(mbrookhart): Relay doesn't currently support convtranspose with dilation
        # verify_convtranspose_with_padding(
        #     (1, 1) + repeat(5, D),
        #     (1, 1) + repeat(3, D),
        #     2 * repeat(2, D),
        #     repeat(3, D),
        #     repeat(1, D),
        #     repeat(2, D),
        # )

    # Convolution with output_shape
    for dims in [1, 2, 3]:
        for num in range(60, 66):
            verify_convtranspose_with_output_shape(
                (1, 1) + repeat(32, dims),
                (1, 1) + repeat(4, dims),
                repeat(num, dims),
                repeat(4, dims),
                repeat(2, dims),
                repeat(1, dims),
            )

            verify_convtranspose_with_output_shape(
                (1, 1) + repeat(32, dims),
                (1, 1) + repeat(4, dims),
                repeat(num, dims),
                repeat(4, dims),
                repeat(2, dims),
                repeat(1, dims),
                auto_pad="SAME_LOWER",
            )

            verify_convtranspose_with_output_shape(
                (1, 1) + repeat(32, dims),
                (1, 2) + repeat(4, dims),
                repeat(num, dims),
                repeat(4, dims),
                repeat(2, dims),
                repeat(1, dims),
                auto_pad="SAME_UPPER",
            )

    verify_convtranspose_with_output_shape(
        (1, 1, 3, 3),
        (1, 2, 3, 3),
        (6, 6),
        (3, 3),
        (2, 2),
        (1, 1),
        auto_pad="SAME_UPPER",
    )

    verify_convtranspose_with_output_shape(
        (1, 1, 3, 3),
        (1, 2, 3, 3),
        (6, 6),
        (3, 3),
        (2, 2),
        (1, 1),
        auto_pad="SAME_LOWER",
    )


@tvm.testing.parametrize_targets
def test_unsqueeze_constant(target, dev):
    """test_unsqueeze_constant"""

    class Flatten(Module):
        def forward(self, input_):
            return input_.view(input_.size(0), -1)

    with tempfile.NamedTemporaryFile() as f:
        file_name = f.name
        input_size = (1, 16, 32, 32)
        dummy_input = torch.randn(*input_size)
        layer = Sequential(Flatten(), Linear(16 * 32 * 32, 64))
        torch.onnx.export(layer, dummy_input, file_name, export_params=True)

        onnx_model = onnx.load(file_name)
        relay.frontend.from_onnx(onnx_model, {"onnx::Reshape_0": input_size})


@tvm.testing.parametrize_targets
def test_pooling(target, dev):
    """test_pooling"""

    def verify_pooling(x_shape, kernel_shape, strides, pads, out_shape, mode, auto_pad="NOTSET"):
        _ = np.random.uniform(size=x_shape).astype("float32")

        if mode == "max":
            node_type = "MaxPool"
        elif mode == "average":
            node_type = "AveragePool"
        else:
            raise ValueError(f"Pool method {mode} is not supported.")

        pool_node = helper.make_node(
            node_type, inputs=["x"], outputs=["y"], kernel_shape=kernel_shape, strides=strides
        )

        if pads is None:
            pad_attr = helper.make_attribute("auto_pad", auto_pad)
        else:
            pad_attr = helper.make_attribute("pads", pads)
        pool_node.attribute.append(pad_attr)

        if mode == "max":
            storage_attr = helper.make_attribute("storage_order", 0)
            pool_node.attribute.append(storage_attr)

        graph = helper.make_graph(
            [pool_node],
            "pooling_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="pooling_test")
        verify_with_ort(
            model,
            [x_shape],
            [out_shape],
            use_vm=False,
            target=target,
            dev=dev,
        )

    for mode in ["max", "average"]:
        # Pool1D
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[1],
            pads=[1, 1],
            out_shape=[1, 1, 32],
            mode=mode,
        )
        # Pool2D
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            out_shape=[1, 1, 32, 32],
            mode=mode,
        )

        # Pool1D with stride
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[2],
            pads=[1, 1],
            out_shape=[1, 1, 16],
            mode=mode,
        )
        # Pool2D with stride
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            out_shape=[1, 1, 16, 16],
            mode=mode,
        )

        # Pool1D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[2],
            pads=None,
            out_shape=[1, 1, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )
        # Pool2D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=None,
            out_shape=[1, 1, 16, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )

        # Pool3D with stride
        verify_pooling(
            x_shape=[1, 1, 32, 32, 32],
            kernel_shape=[3, 3, 3],
            strides=[2, 2, 2],
            pads=[1, 1, 1, 1, 1, 1],
            out_shape=[1, 1, 16, 16, 16],
            mode=mode,
        )

        # Pool3D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32, 32, 32],
            kernel_shape=[3, 3, 3],
            strides=[2, 2, 2],
            pads=None,
            out_shape=[1, 1, 16, 16, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )


@tvm.testing.parametrize_targets
def test_global_pooling(target, dev):
    """test_global_pooling"""

    def verify_global_pooling(x_shape, mode):
        out_shape = x_shape[:2] + [1] * (len(x_shape) - 2)

        if mode == "max":
            node_type = "GlobalMaxPool"
        elif mode == "average":
            node_type = "GlobalAveragePool"
        else:
            raise ValueError(f"Pool method {mode} is not supported.")

        pool_node = helper.make_node(node_type, inputs=["x"], outputs=["y"])

        graph = helper.make_graph(
            [pool_node],
            "global_pooling_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="global_pooling_test")
        verify_with_ort(
            model,
            [x_shape],
            [out_shape],
            use_vm=False,
            target=target,
            dev=dev,
        )

    # Test each pooling mode across all N-D inputs.
    for mode in ["average", "max"]:
        # 1D Pooling (NCW)
        verify_global_pooling([1, 8, 8], mode)
        verify_global_pooling([4, 1, 4], mode)
        # 2D Pooling (NCHW)
        verify_global_pooling([1, 8, 8, 8], mode)
        verify_global_pooling([4, 1, 6, 4], mode)
        # 3D Pooling (NCDHW)
        verify_global_pooling([1, 8, 6, 8, 8], mode)
        verify_global_pooling([4, 1, 2, 6, 4], mode)


@pytest.mark.skip("flaky")
@tvm.testing.parametrize_targets
def test_qlinear_average_pool(target, dev):
    """test_qlinear_average_pool"""

    def verify_qlinear_average_pool(
        x_shape, kernel_shape, strides, pads, out_shape, auto_pad="NOTSET"
    ):
        input_nodes = [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape)),
        ]

        output_nodes = [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(out_shape)),
        ]

        input_names = ["X"]

        node = helper.make_node(
            "AveragePool",
            inputs=input_names,
            outputs=["Y"],
            kernel_shape=kernel_shape,
            strides=strides,
        )

        if pads is None:
            pad_attr = helper.make_attribute("auto_pad", auto_pad)
        else:
            pad_attr = helper.make_attribute("pads", pads)
        node.attribute.append(pad_attr)

        graph = helper.make_graph(
            [node],
            "qlinear_average_pool_test",
            inputs=input_nodes,
            outputs=output_nodes,
        )

        model = helper.make_model(graph, producer_name="qlinear_average_pool_Test")
        quantize_and_verify_with_ort(model, input_names, [x_shape], target, dev)

    # Pool1D
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32],
        kernel_shape=[3],
        strides=[1],
        pads=[1, 1],
        out_shape=[1, 1, 32],
    )
    # Pool2D
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 32, 32],
    )

    # Pool1D with stride
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32],
        kernel_shape=[3],
        strides=[2],
        pads=[1, 1],
        out_shape=[1, 1, 16],
    )
    # Pool2D with stride
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 16, 16],
    )

    # Pool1D with stride and autopadding
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32],
        kernel_shape=[3],
        strides=[2],
        pads=None,
        out_shape=[1, 1, 16],
        auto_pad="SAME_UPPER",
    )
    # Pool2D with stride and autopadding
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool3D with stride
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        strides=[2, 2, 2],
        pads=[1, 1, 1, 1, 1, 1],
        out_shape=[1, 1, 16, 16, 16],
    )

    # Pool3D with stride and autopadding
    verify_qlinear_average_pool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        strides=[2, 2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16, 16],
        auto_pad="SAME_UPPER",
    )


@tvm.testing.parametrize_targets
def test_qlinear_global_average_pool(target, dev):
    """test_qlinear_global_average_pool"""

    def verify_qlinear_global_average_pool(x_shape):
        out_shape = x_shape[:2] + [1] * (len(x_shape) - 2)

        node_type = "GlobalAveragePool"

        input_names = ["X"]

        pool_node = helper.make_node(node_type, inputs=input_names, outputs=["Y"])

        graph = helper.make_graph(
            [pool_node],
            "qlinear_global_average_pool_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="qlinear_global_average_pool_test")
        quantize_and_verify_with_ort(model, input_names, [x_shape], target, dev)

    # 1D Pooling (NCW)
    verify_qlinear_global_average_pool([1, 8, 8])
    verify_qlinear_global_average_pool([4, 1, 4])

    # 2D Pooling (NCHW)
    verify_qlinear_global_average_pool([1, 8, 8, 8])
    verify_qlinear_global_average_pool([4, 1, 6, 4])

    # 3D Pooling (NCDHW)
    verify_qlinear_global_average_pool([1, 8, 6, 8, 8])
    verify_qlinear_global_average_pool([4, 1, 2, 6, 4])


@tvm.testing.parametrize_targets
def test_mod(target, dev):
    """test_mod"""

    def verify_mod(x_shape, y_shape, fmod, out_shape, dtype="float32"):
        x_np = np.random.uniform(-100.0, 100.0, x_shape).astype(dtype)
        y_np = np.random.uniform(-100.0, 100.0, y_shape).astype(dtype)
        y_np = np.where(y_np == 0, 1, y_np)  # remove 0's to avoid division by zero error

        mod_node = helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=fmod)

        onnx_dtype = TensorProto.FLOAT if dtype == "float32" else TensorProto.INT32
        graph = helper.make_graph(
            [mod_node],
            "mod_test",
            inputs=[
                helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
                helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
            ],
            outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="mod_test")
        verify_with_ort_with_inputs(model, [x_np, y_np], [out_shape], target=target, dev=dev)

    # Mod
    verify_mod(
        x_shape=[1, 32, 32], y_shape=[1, 1, 32], fmod=0, out_shape=(1, 32, 32), dtype="int32"
    )
    verify_mod(
        x_shape=[1, 32, 32, 32],
        y_shape=[1, 32, 32, 32],
        fmod=0,
        out_shape=(1, 32, 32, 32),
        dtype="int32",
    )

    # fmod
    verify_mod(
        x_shape=[1, 32, 32], y_shape=[1, 32, 32], fmod=1, out_shape=(1, 32, 32), dtype="int32"
    )
    verify_mod(x_shape=[1, 1, 32, 32], y_shape=[1, 32, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))
    verify_mod(x_shape=[1, 32, 32, 32], y_shape=[1, 1, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))
    verify_mod(
        x_shape=[1, 32, 32, 32],
        y_shape=[1, 32, 32, 32],
        fmod=1,
        out_shape=(1, 32, 32, 32),
        dtype="int32",
    )
    verify_mod(x_shape=[1, 32, 32, 32], y_shape=[1, 32, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))


@tvm.testing.parametrize_targets
def test_xor(target, dev):
    """test_xor"""

    def verify_xor(x_shape, y_shape):
        x_np = np.random.choice(a=[False, True], size=x_shape).astype("bool")
        y_np = np.random.choice(a=[False, True], size=y_shape).astype("bool")

        np_out = np.logical_xor(x_np, y_np)
        out_shape = np_out.shape

        xor_node = helper.make_node("Xor", inputs=["x", "y"], outputs=["z"])

        onnx_dtype = TensorProto.BOOL
        graph = helper.make_graph(
            [xor_node],
            "xor_test",
            inputs=[
                helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
                helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
            ],
            outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="xor_test")
        verify_with_ort_with_inputs(model, [x_np, y_np], [out_shape], target=target, dev=dev)

    # XOR
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 32, 32])

    # Xor broadcast
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 1, 32])


@tvm.testing.parametrize_targets
def test_max_roi_pool(target, dev):
    """test_max_roi_pool"""

    def verify_max_roi_pool(x_shape, rois_shape, pooled_shape, spatial_scale, out_shape):
        if spatial_scale is None:
            pool_node = helper.make_node(
                "MaxRoiPool", inputs=["x", "rois"], outputs=["y"], pooled_shape=pooled_shape
            )
        else:
            pool_node = helper.make_node(
                "MaxRoiPool",
                inputs=["x", "rois"],
                outputs=["y"],
                pooled_shape=pooled_shape,
                spatial_scale=spatial_scale,
            )

        graph = helper.make_graph(
            [pool_node],
            "pool_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(rois_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="pool_test")
        verify_with_ort(model, [x_shape, rois_shape], [out_shape], target=target, dev=dev)

    verify_max_roi_pool(
        x_shape=[1, 3, 6, 6],
        rois_shape=[3, 5],
        pooled_shape=[1, 1],
        spatial_scale=None,
        out_shape=[3, 3, 1, 1],
    )

    verify_max_roi_pool(
        x_shape=[1, 3, 10, 10],
        rois_shape=[4, 5],
        pooled_shape=[2, 2],
        spatial_scale=2.0,
        out_shape=[4, 3, 2, 2],
    )


@tvm.testing.parametrize_targets
def test_lppool(target, dev):
    """test_lppool"""

    def verify_lppool(x_shape, kernel_shape, p, strides, pads, out_shape, auto_pad="NOTSET"):
        kwargs = {}
        if p is not None:
            kwargs["p"] = p
        if pads is None:
            pool_node = helper.make_node(
                "LpPool",
                inputs=["x"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                auto_pad=auto_pad,
                strides=strides,
                **kwargs,
            )
        else:
            pool_node = helper.make_node(
                "LpPool",
                inputs=["x"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                **kwargs,
            )

        graph = helper.make_graph(
            [pool_node],
            "lppool_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="lppool_test")
        verify_with_ort(
            model,
            [x_shape],
            [out_shape],
            use_vm=True,
            target=target,
            dev=dev,
        )

    # Pool1D
    verify_lppool(
        x_shape=[1, 1, 32], kernel_shape=[3], p=2, strides=[1], pads=[1, 1], out_shape=[1, 1, 32]
    )

    # Pool2D
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 32, 32],
    )

    # Pool1D with stride
    verify_lppool(
        x_shape=[1, 1, 32], kernel_shape=[3], p=2, strides=[2], pads=[1, 1], out_shape=[1, 1, 16]
    )

    # Pool2D with stride
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 16, 16],
    )

    # Pool1D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32],
        kernel_shape=[3],
        p=2,
        strides=[2],
        pads=None,
        out_shape=[1, 1, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool2D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool2D with empty stride
    verify_lppool(
        x_shape=[1, 3, 32, 32],
        kernel_shape=[2, 2],
        p=4,
        strides=None,
        pads=None,
        out_shape=[1, 3, 32, 32],
        auto_pad="SAME_LOWER",
    )

    # Pool3D with stride
    verify_lppool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        p=2,
        strides=[2, 2, 2],
        pads=[1, 1, 1, 1, 1, 1],
        out_shape=[1, 1, 16, 16, 16],
    )

    # Pool3D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        p=2,
        strides=[2, 2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16, 16],
        auto_pad="SAME_UPPER",
    )
    # Pool2D with empty p
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=None,
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 32, 32],
    )


def verify_global_lppool(x_shape, p, out_shape, target, dev):
    """verify_global_lppool"""
    pool_node = helper.make_node(
        "GlobalLpPool",
        inputs=["x"],
        outputs=["y"],
        p=p,
    )

    graph = helper.make_graph(
        [pool_node],
        "global_lppool_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="global_lppool_test")
    verify_with_ort(model, [x_shape], out_shape, use_vm=True, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_global_lppool(target, dev):
    """test_global_lppool"""
    # LpPool1D
    verify_global_lppool(x_shape=[1, 15, 16], p=2, out_shape=[1, 15, 1], target=target, dev=dev)

    # LpPool2D
    verify_global_lppool(
        x_shape=[1, 15, 32, 32], p=2, out_shape=[1, 15, 1, 1], target=target, dev=dev
    )

    # LpPool2D
    verify_global_lppool(
        x_shape=[1, 15, 32, 32], p=3, out_shape=[1, 15, 1, 1], target=target, dev=dev
    )

    # LpPool3D
    verify_global_lppool(
        x_shape=[1, 15, 3, 32, 32], p=2, out_shape=[1, 15, 1, 1, 1], target=target, dev=dev
    )


def verify_rnn(
    seq_length,
    batch_size,
    input_size,
    hidden_size,
    rnn_type="LSTM",
    use_bias=False,
    activations=None,
    alphas=None,
    betas=None,
    use_initial_state=False,
    use_peep=False,
    linear_before_reset=False,
    directions=1,
    layout=0,
    rtol=1e-5,
    atol=1e-5,
    target=None,
    dev=None,
    use_sequence_lens=False,
):
    """verify_rnn"""
    if rnn_type == "RNN":
        multiplier = 1
    elif rnn_type == "LSTM":
        multiplier = 4
    elif rnn_type == "GRU":
        multiplier = 3
    else:
        raise NotImplementedError(f"{rnn_type} RNNs not yet supported.")

    if directions not in [1, 2]:
        raise ValueError(f"Direction should be either 1 or 2 (for bidirectional LSTMs)")

    def get_inputs():
        input_names = []
        input_values = []
        input_tensors = []

        def register(np_arr, name, shape=None):
            input_values.append(np_arr)
            input_names.append(name)

            # Map of numpy dtypes to the protobuf equivalent
            dtype_map = {
                "float32": TensorProto.FLOAT,
                "int32": TensorProto.INT32,
                "int8": TensorProto.INT8,
            }

            if np_arr.dtype.name not in dtype_map:
                raise ValueError(f"Unknown dtype we don't know how to handle {np.dtype.name}")
            if shape is None:
                shape = list(np_arr.shape)
            proto_type = dtype_map[np_arr.dtype.name]
            input_tensors.append(helper.make_tensor_value_info(name, proto_type, shape))

        if layout == 1:
            x_np = np.random.uniform(size=(batch_size, seq_length, input_size)).astype("float32")
        else:
            x_np = np.random.uniform(size=(seq_length, batch_size, input_size)).astype("float32")
        w_np = np.random.uniform(size=(directions, multiplier * hidden_size, input_size)).astype(
            "float32"
        )
        r_np = np.random.uniform(size=(directions, multiplier * hidden_size, hidden_size)).astype(
            "float32"
        )
        register(x_np, "X")
        register(w_np, "W")
        register(r_np, "R")

        if use_bias:
            b_np = np.random.uniform(size=(directions, multiplier * 2 * hidden_size)).astype(
                "float32"
            )
            register(b_np, "B")

        if use_sequence_lens:
            sequence_np = np.random.uniform(0, seq_length, size=(batch_size)).astype("int32")
            register(sequence_np, "sequence_lens")

        if use_initial_state:
            assert use_bias is True, "Initial states must have bias specified."

            if not use_sequence_lens:
                sequence_np = np.repeat(seq_length, batch_size).astype("int32")
                register(sequence_np, "sequence_lens")

            if layout == 1:
                initial_h_np = np.random.uniform(size=(batch_size, directions, hidden_size)).astype(
                    "float32"
                )
            else:
                initial_h_np = np.random.uniform(size=(directions, batch_size, hidden_size)).astype(
                    "float32"
                )
            register(initial_h_np, "initial_h")

            if rnn_type == "LSTM":
                if layout == 1:
                    initial_c_np = np.random.uniform(
                        size=(batch_size, directions, hidden_size)
                    ).astype("float32")
                else:
                    initial_c_np = np.random.uniform(
                        size=(directions, batch_size, hidden_size)
                    ).astype("float32")
                register(initial_c_np, "initial_c")

        if use_peep and rnn_type == "LSTM":
            assert use_initial_state is True, "Peepholes require initial state to be specified."
            p_np = np.random.uniform(size=(directions, 3 * hidden_size)).astype("float32")
            register(p_np, "P")

        return input_names, input_tensors, input_values

    input_names, input_tensors, input_values = get_inputs()

    def get_outputs():
        output_names = []
        graph_outputs = []
        output_shapes = []

        def register(name, shape, proto_type):
            output_names.append(name)
            graph_outputs.append(helper.make_tensor_value_info(name, proto_type, list(shape)))
            output_shapes.append(list(shape))

        if layout == 1:
            register("Y", [directions, seq_length, batch_size, hidden_size], TensorProto.FLOAT)
            register("Y_h", [batch_size, directions, hidden_size], TensorProto.FLOAT)
        else:
            register("Y", [seq_length, directions, batch_size, hidden_size], TensorProto.FLOAT)
            register("Y_h", [directions, batch_size, hidden_size], TensorProto.FLOAT)

        if rnn_type == "LSTM":
            if layout == 1:
                register("Y_c", [batch_size, directions, hidden_size], TensorProto.FLOAT)
            else:
                register("Y_c", [directions, batch_size, hidden_size], TensorProto.FLOAT)

        return output_names, graph_outputs, output_shapes

    output_names, graph_outputs, output_shapes = get_outputs()

    rnn_node = helper.make_node(
        rnn_type, inputs=input_names, outputs=output_names, hidden_size=hidden_size
    )
    if activations is not None:
        activations_attr = helper.make_attribute("activations", activations)
        rnn_node.attribute.append(activations_attr)
    if directions == 2:
        direction_attr = helper.make_attribute("direction", "bidirectional")
        rnn_node.attribute.append(direction_attr)
    if alphas is not None:
        alphas_attr = helper.make_attribute("activation_alpha", alphas)
        rnn_node.attribute.append(alphas_attr)
    if betas is not None:
        betas_attr = helper.make_attribute("activation_beta", betas)
        rnn_node.attribute.append(betas_attr)
    if linear_before_reset and rnn_type == "GRU":
        lbr_attr = helper.make_attribute("linear_before_reset", 1)
        rnn_node.attribute.append(lbr_attr)
    if layout == 1:
        layout_attr = helper.make_attribute("layout", 1)
        rnn_node.attribute.append(layout_attr)

    graph = helper.make_graph([rnn_node], "rnn_test", inputs=input_tensors, outputs=graph_outputs)

    model = helper.make_model(graph, producer_name="rnn_test")

    verify_with_ort_with_inputs(
        model, input_values, output_shapes, atol=atol, rtol=rtol, target=target, dev=dev
    )


def verify_rnn_helper(target, dev, rnn_type):
    num_activations = 1
    if rnn_type == "GRU":
        num_activations = 2
    elif rnn_type == "LSTM":
        num_activations = 3

    for directions in [1, 2]:
        # No bias.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # large batch.
        verify_rnn(
            seq_length=4,
            batch_size=8,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Non power of two.
        verify_rnn(
            seq_length=3,
            batch_size=3,
            input_size=16,
            hidden_size=40,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Long sequence.
        verify_rnn(
            seq_length=8,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Large hidden.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=128,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Large input.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=64,
            hidden_size=32,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )

        # Different activation testing.
        # Default value hardsigmoid.
        # TODO: onnxruntime <= v1.12.0 has wrong default value of all activation functions
        if rnn_type != "RNN":
            activations = ["HardSigmoid", "Tanh", "Tanh"][0:num_activations] * directions
            verify_rnn(
                seq_length=2,
                batch_size=1,
                input_size=16,
                hidden_size=32,
                use_bias=False,
                activations=activations,
                rnn_type=rnn_type,
                directions=directions,
                target=target,
                dev=dev,
            )
        # Multiple parametrized activations.
        activations = ["HardSigmoid", "LeakyRelu", "Tanh"][0:num_activations] * directions
        alphas = [2.0, 0.5, 0.0][0:num_activations] * directions
        betas = [0.3, 0.0, 0.0][0:num_activations] * directions
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=activations,
            alphas=alphas,
            betas=betas,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # All parametrized with new Affine activation.
        activations = ["Affine", "LeakyRelu", "HardSigmoid"][0:num_activations] * directions
        alphas = [0.8, 2.0, 0.5][0:num_activations] * directions
        betas = [0.0, 0.3, 0.0][0:num_activations] * directions
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=activations,
            alphas=alphas,
            betas=betas,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )

        # Testing with initial state
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            use_initial_state=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )

        # Testing layout
        # TODO: onnxruntime <= 1.12.0 doesn't support layout == 1
        # verify_rnn(
        #     seq_length=2,
        #     batch_size=1,
        #     input_size=16,
        #     hidden_size=32,
        #     use_bias=True,
        #     rnn_type="RNN",
        #     directions=directions,
        #     layout=1,
        #     target=target,
        #     dev=dev,
        # )

        # Testing with initial state
        if rnn_type == "GRU":
            verify_rnn(
                seq_length=2,
                batch_size=1,
                input_size=16,
                hidden_size=32,
                use_bias=True,
                use_initial_state=True,
                rnn_type=rnn_type,
                directions=directions,
                target=target,
                dev=dev,
                use_sequence_lens=True,
            )
            verify_rnn(
                seq_length=8,
                batch_size=8,
                input_size=16,
                hidden_size=32,
                use_bias=True,
                use_initial_state=True,
                rnn_type=rnn_type,
                directions=directions,
                target=target,
                dev=dev,
                use_sequence_lens=True,
            )

        # Testing with peepholes
        if rnn_type == "LSTM":
            verify_rnn(
                seq_length=2,
                batch_size=1,
                input_size=16,
                hidden_size=32,
                use_bias=True,
                use_initial_state=True,
                use_peep=True,
                rnn_type="LSTM",
                directions=directions,
                target=target,
                dev=dev,
            )


@tvm.testing.parametrize_targets
def test_rnn(target, dev):
    verify_rnn_helper(target, dev, "RNN")


@tvm.testing.parametrize_targets
def test_lstm(target, dev):
    verify_rnn_helper(target, dev, "LSTM")


@tvm.testing.parametrize_targets
def test_gru(target, dev):
    verify_rnn_helper(target, dev, "GRU")


@tvm.testing.parametrize_targets
def test_resize(target, dev):
    """test_resize"""

    def verify(ishape, oshape, scales, mode, coord_trans="asymmetric", alpha=0.5, exclude=False):
        nodes = [
            make_constant_node("roi", onnx.TensorProto.FLOAT, (0,), []),
            make_constant_node("scales", onnx.TensorProto.FLOAT, (len(scales),), scales),
        ]
        input_names = ["X", "roi", "scales"]

        if oshape != []:
            nodes.append(
                make_constant_node("sizes", onnx.TensorProto.INT64, (len(oshape),), oshape)
            )
            input_names.append("sizes")
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=input_names,
                outputs=["Y"],
                mode=mode,
                coordinate_transformation_mode=coord_trans,
                cubic_coeff_a=alpha,
                exclude_outside=exclude,
            )
        )

        if oshape == []:
            oshape = [round(dim * scale) for (dim, scale) in zip(ishape, scales)]
        graph = helper.make_graph(
            nodes,
            "resize_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, ishape)],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, oshape)],
        )

        model = helper.make_model(graph, producer_name="resize_test")

        verify_with_ort(
            model,
            [ishape],
            [oshape],
            use_vm=True,
            opset=11,
            freeze_params=True,
            target=target,
            dev=dev,
        )

    for ndim in [1, 2, 3]:
        method = "nearest"
        for coord_trans in ["asymmetric", "align_corners", "half_pixel"]:
            # upsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [64] * ndim, [], method, coord_trans)
            # downsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [16] * ndim, [], method, coord_trans)
            # scales are specified instead of sizes
            verify([1, 16] + [32] * ndim, [], [1, 1] + [0.5] * ndim, method, coord_trans)
            verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, method, coord_trans)
            verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, None, coord_trans)

        method = "linear"
        # upsampling
        verify([1, 16] + [32] * ndim, [1, 16] + [64] * ndim, [], method)
        # downsampling
        verify([1, 16] + [32] * ndim, [1, 16] + [16] * ndim, [], method)
        # scales are specified instead of sizes
        verify([1, 16] + [32] * ndim, [], [1, 1] + [0.5] * ndim, method)
        verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, method)

        if ndim == 2:
            # ONNX Runtime only supports cubic interpolation for 2D images
            method = "cubic"
            for alpha in [0.5, 0.75]:
                for exclude in [True, False]:
                    # upsampling
                    verify(
                        [1, 16] + [32] * ndim,
                        [1, 16] + [64] * ndim,
                        [],
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    # downsampling
                    verify(
                        [1, 16] + [32] * ndim,
                        [1, 16] + [16] * ndim,
                        [],
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    # scales are specified instead of sizes
                    verify(
                        [1, 16] + [32] * ndim,
                        [],
                        [1, 1] + [0.5] * ndim,
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    verify(
                        [1, 16] + [32] * ndim,
                        [],
                        [1, 1] + [2] * ndim,
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )

    def verify_opset_10(ishape, scales, mode):
        nodes = [
            make_constant_node("scales", onnx.TensorProto.FLOAT, (len(scales),), scales),
        ]
        input_names = ["X", "scales"]
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=input_names,
                outputs=["Y"],
                mode=mode,
            )
        )

        oshape = [round(dim * scale) for (dim, scale) in zip(ishape, scales)]
        graph = helper.make_graph(
            nodes,
            "resize_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, ishape)],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, oshape)],
        )

        model = helper.make_model(graph, producer_name="resize_test")
        verify_with_ort(
            model,
            [ishape],
            [oshape],
            use_vm=True,
            freeze_params=True,
            opset=10,
            target=target,
            dev=dev,
        )

    verify_opset_10([1, 16, 32, 32], [1, 1, 2, 2], "nearest")
    verify_opset_10([1, 16, 32, 32], [1, 1, 0.5, 0.5], "linear")


@tvm.testing.parametrize_targets
def test_nonzero(target, dev):
    """test_nonzero"""

    def verify_nonzero(indata, outdata, dtype):
        node = helper.make_node(
            "NonZero",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "nonzero_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="nonzero_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype="int64", use_vm=True, opset=9, target=target, dev=dev
        )

    input_data = np.array([[1, 0], [1, 1]], dtype=np.int64)
    result = np.array((np.nonzero(input_data)))  # expected output [[0, 1, 1], [0, 0, 1]]
    verify_nonzero(input_data, result, dtype=np.int64)

    input_data = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.int64)
    result = np.array((np.nonzero(input_data)))  # expected output [[0, 1, 2, 2], [0, 1, 0, 1]]
    verify_nonzero(input_data, result, dtype=np.int64)


@tvm.testing.parametrize_targets
def test_topk(target, dev):
    """test_topk"""

    def verify_topk(input_dims, k, axis=-1):
        output_dims = list(input_dims)
        output_dims[axis] = k

        node = helper.make_node("TopK", inputs=["X", "K"], outputs=["Values", "Indices"], axis=axis)

        graph = helper.make_graph(
            [node],
            "topk_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_dims)),
                helper.make_tensor_value_info(
                    "K",
                    TensorProto.INT64,
                    [
                        1,
                    ],
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("Values", TensorProto.FLOAT, output_dims),
                helper.make_tensor_value_info("Indices", TensorProto.INT64, output_dims),
            ],
        )

        model = helper.make_model(graph, producer_name="topk_test")

        indata = np.random.uniform(-10, 10, input_dims).astype(np.float32)
        verify_with_ort_with_inputs(
            model, [indata, np.array([k])], use_vm=True, target=target, dev=dev
        )

    for n in [12, 32]:
        for shape in [[n], [n, n], [n, n, n]]:
            for k in [1, 5, 10]:
                verify_topk(shape, k)

        verify_topk([n, n, n], 5, 0)
        verify_topk([n, n, n], 5, 1)
        verify_topk([n, n, n], 5, 2)


@tvm.testing.parametrize_targets
def test_roi_align(target, dev):
    """test_roi_align"""

    def verify_roi_align(
        input_dims,
        num_roi,
        output_height,
        output_width,
        sampling_ratio=0,
        spatial_scale=1.0,
        mode="avg",
    ):
        output_dims = [num_roi, input_dims[1], output_height, output_width]

        node = helper.make_node(
            "RoiAlign",
            coordinate_transformation_mode="output_half_pixel",
            inputs=["X", "rois", "batch_indices"],
            outputs=["Y"],
            mode=mode,
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        )

        graph = helper.make_graph(
            [node],
            "roialign_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_dims)),
                helper.make_tensor_value_info("rois", TensorProto.FLOAT, [num_roi, 4]),
                helper.make_tensor_value_info(
                    "batch_indices",
                    TensorProto.INT64,
                    [
                        num_roi,
                    ],
                ),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_dims)],
        )

        model = helper.make_model(graph, producer_name="roialign_test")

        np_data = np.random.uniform(size=input_dims).astype("float32")
        np_rois = np.random.uniform(size=[num_roi, 4]).astype("float32") * input_dims[2]
        np_batch_indices = np.random.randint(low=0, high=input_dims[0], size=num_roi)

        verify_with_ort_with_inputs(
            model,
            [np_data, np_rois, np_batch_indices],
            out_shape=[output_dims],
            target=target,
            dev=dev,
        )

    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((4, 4, 16, 32), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 8, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 8, 8), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 16, 5, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 12), 8, 7, 3, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=0.5)
    verify_roi_align((3, 4, 12, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.5)
    verify_roi_align((5, 4, 16, 14), 32, 7, 7, sampling_ratio=1, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=2, spatial_scale=1.0)

    # ONNX implementation of roi_align with max mode is incorrect, so we don't compare outputs here.


@tvm.testing.parametrize_targets
def test_non_max_suppression(target, dev):
    """test_non_max_suppression"""

    def verify_nms(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_dims
    ):
        input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold"]
        input_nodes = [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes.shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores.shape),
            helper.make_tensor_value_info(
                "max_output_boxes_per_class", TensorProto.INT64, max_output_boxes_per_class.shape
            ),
            helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, iou_threshold.shape),
        ]
        inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold]
        if score_threshold is not None:
            input_names.append("score_threshold")
            input_nodes.append(
                helper.make_tensor_value_info(
                    "score_threshold", TensorProto.FLOAT, score_threshold.shape
                )
            )
            inputs.append(score_threshold)
        node = helper.make_node(
            "NonMaxSuppression",
            inputs=input_names,
            outputs=["Y"],
            center_point_box=0,
        )

        graph = helper.make_graph(
            [node],
            "nms_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, output_dims)],
        )

        model = helper.make_model(graph, producer_name="nms_test")

        verify_with_ort_with_inputs(model, inputs, use_vm=True, target=target, dev=dev)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.9, 0.9],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.95, 0.95],
                [0.5, 0.5, 0.96, 0.96],
                [0.5, 0.5, 1.0, 1.0],
            ],
        ]
    ).astype("float32")

    scores = np.array(
        [
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
        ]
    ).astype("float32")
    max_output_boxes_per_class = np.array(2).astype("int64")
    iou_threshold = np.array(0.8).astype("float32")
    output_dims = [8, 3]
    verify_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, None, output_dims)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)
    output_dims = [2, 3]
    verify_nms(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_dims
    )


@tvm.testing.parametrize_targets
def test_loop(target, dev):
    """test_loop"""

    def verify_cond_loop():
        y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [1])
        y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [1])
        scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

        y = np.array([-2]).astype(np.float32)

        five_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["five"],
            value=helper.make_tensor(
                name="const_tensor_five", data_type=TensorProto.FLOAT, dims=(), vals=[5]
            ),
        )

        iter_cast_node = helper.make_node(
            "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
        )

        y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

        less_node = helper.make_node("Less", inputs=["y_out", "five"], outputs=["cond_less"])

        squeeze_node = helper.make_node("Squeeze", inputs=["cond_less"], outputs=["cond_squeeze"])

        cond_cast_node = helper.make_node(
            "Cast", inputs=["cond_squeeze"], outputs=["cond_out"], to=onnx.TensorProto.BOOL
        )

        scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

        loop_body = helper.make_graph(
            [
                five_const_node,
                iter_cast_node,
                y_add_node,
                less_node,
                squeeze_node,
                cond_cast_node,
                scan_identity_node,
            ],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        loop_node = helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        _ = np.array([13]).astype(np.float32)
        cond = np.array(1).astype(bool)
        loop_graph = onnx.helper.make_graph(
            [loop_node],
            "loop_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1]),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5, 1]),
            ],
        )
        loop_model = onnx.helper.make_model(loop_graph)

        # Set a high trip count so that condition trips first.
        trip_count = np.array(40).astype(np.int64)
        cond = np.array(1).astype(bool)
        input_vals = [trip_count, cond, y]
        verify_with_ort_with_inputs(
            loop_model,
            input_vals,
            use_vm=True,
            freeze_params=True,
            opset=11,
            target=target,
            dev=dev,
        )

    def verify_count_loop():
        y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [])
        y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [])
        scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

        y = np.array(-2).astype(np.float32)

        iter_cast_node = helper.make_node(
            "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
        )

        y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

        identity_node = helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

        scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

        loop_body = helper.make_graph(
            [identity_node, iter_cast_node, y_add_node, scan_identity_node],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        loop_node = helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        _ = np.array([13]).astype(np.float32)
        cond = np.array(1).astype(bool)
        loop_graph = onnx.helper.make_graph(
            [loop_node],
            "loop_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, []),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5]),
            ],
        )
        loop_model = onnx.helper.make_model(loop_graph)

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)
        input_vals = [trip_count, cond, y]
        verify_with_ort_with_inputs(
            loop_model,
            input_vals,
            use_vm=True,
            freeze_params=True,
            opset=11,
            target=target,
            dev=dev,
        )

    def verify_tensor_loop(shapeless_output=False):
        y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [3, 3, 3, 3])
        y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [3, 3, 3, 3])
        scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [3, 3, 3, 3])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

        y = np.random.normal(size=[3, 3, 3, 3]).astype(np.float32)

        iter_cast_node = helper.make_node(
            "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
        )

        y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

        identity_node = helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

        scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

        loop_body = helper.make_graph(
            [identity_node, iter_cast_node, y_add_node, scan_identity_node],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        loop_node = helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)

        # Allow testing of malformed nodes since pytorch likes to create these.
        if shapeless_output:
            scan_shape = None
        else:
            scan_shape = [5, 3, 3, 3, 3]

        loop_graph = onnx.helper.make_graph(
            [loop_node],
            "loop_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 3, 3, 3]),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [3, 3, 3, 3]),
                onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, scan_shape),
            ],
        )
        loop_model = onnx.helper.make_model(loop_graph)

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)
        input_vals = [trip_count, cond, y]
        verify_with_ort_with_inputs(
            loop_model,
            input_vals,
            use_vm=True,
            freeze_params=True,
            opset=11,
            target=target,
            dev=dev,
        )

    # Test a loop that exits once a condition is met.
    verify_cond_loop()
    # Test a loop that exits after a fixed number of iterations with scalar outputs.
    verify_count_loop()
    # Test a loop that uses an array output.
    verify_tensor_loop()
    # Test a loop that is malformed and has no output shape defined.
    verify_tensor_loop(shapeless_output=True)


@tvm.testing.parametrize_targets
def test_if(target, dev):
    """test_if"""

    def verify_if(cond_array, num_outputs):
        # Given a bool scalar input cond.
        # return constant tensor x if cond is True, otherwise return constant tensor y.

        def append_constant_nodes(nodes, outputs, expected, name):
            outputs.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [5]))

            expected.append(np.random.randn(5).astype("float32"))

            nodes.append(
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[name],
                    value=numpy_helper.from_array(expected[-1]),
                )
            )

        if_outputs = []
        graph_outputs = []

        then_nodes, then_outs, then_expected = [], [], []
        else_nodes, else_outs, else_expected = [], [], []

        for i in range(num_outputs):
            append_constant_nodes(then_nodes, then_outs, then_expected, f"then_out{i}")
            append_constant_nodes(else_nodes, else_outs, else_expected, f"else_out{i}")

            if_outputs.append(f"res{i}")
            graph_outputs.append(
                onnx.helper.make_tensor_value_info(f"res{i}", onnx.TensorProto.FLOAT, [5]),
            )

        then_body = onnx.helper.make_graph(then_nodes, "then_body", [], then_outs)
        else_body = onnx.helper.make_graph(else_nodes, "else_body", [], else_outs)

        if_node = onnx.helper.make_node(
            "If", inputs=["cond"], outputs=if_outputs, then_branch=then_body, else_branch=else_body
        )

        if_graph = onnx.helper.make_graph(
            [if_node],
            "if_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            ],
            outputs=graph_outputs,
        )

        if_model = onnx.helper.make_model(if_graph)
        if cond_array:
            cond = np.array([1]).astype("bool")
        else:
            cond = np.array(1).astype("bool")
        correct_out = then_expected if cond else else_expected

        # TODO(jwfromm): Onnxruntime 1.0.0 is buggy with If statements. Replace this with
        # verify_with_ort once we update versions.
        tvm_out = get_tinygrad_output_with_vm(if_model, [cond], target, dev, freeze_params=True)
        if not isinstance(tvm_out, list):
            tvm_out = [tvm_out]
        for i, _ in enumerate(tvm_out):
            tvm.testing.assert_allclose(
                correct_out[i],
                tvm_out[i],  # pylint: disable=unnecessary-list-index-lookup
                rtol=1e-05,
                atol=1e-05,
            )

    # Confirm that if works with cond as an array or scalar.
    verify_if(cond_array=False, num_outputs=1)
    verify_if(cond_array=False, num_outputs=2)
    verify_if(cond_array=True, num_outputs=1)
    verify_if(cond_array=True, num_outputs=2)


@tvm.testing.parametrize_targets
def test_graph_input_use_in_if(target, dev):
    """test_graph_input_use_in_if"""

    def verify_if(num_nested, cond):
        # return "graph input" if cond is True, else return constant(-1).

        input_tensor = helper.make_tensor_value_info("graph_input", TensorProto.FLOAT, [1])
        output_tensor = helper.make_tensor_value_info("graph_output", TensorProto.FLOAT, [1])
        constant_node = make_constant_node("const_val", TensorProto.FLOAT, [1], [-1])
        cond_tensor = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])
        inner_if_node = None
        for i in range(num_nested):
            identity_node = helper.make_node(
                "Identity",
                inputs=["const_val"],
                outputs=[f"const{i}"],
                name=f"depth{i}'th else identity",
            )
            else_branch = helper.make_graph(
                [identity_node],
                f"else{i}_body",
                inputs=[],
                outputs=[helper.make_tensor_value_info(f"const{i}", TensorProto.FLOAT, [1])],
            )
            out_name = f"if_output{i}" if i != (num_nested - 1) else "graph_output"

            if i == 0:
                identity_node = helper.make_node(
                    "Identity",
                    inputs=["graph_input"],
                    outputs=[f"input_identity{i}"],
                    name=f"depth{i}'th then identity",
                )
                then_branch = helper.make_graph(
                    [identity_node],
                    f"then{i}_body",
                    inputs=[],
                    outputs=[
                        helper.make_tensor_value_info(f"input_identity{i}", TensorProto.FLOAT, [1])
                    ],
                )
                if_node = helper.make_node(
                    "If",
                    inputs=["cond"],
                    outputs=[out_name],
                    then_branch=then_branch,
                    else_branch=else_branch,
                    name=f"depth{i}'s If node",
                )
                inner_if_node = if_node
            else:
                then_branch = helper.make_graph(
                    [inner_if_node],
                    f"then{i}_body",
                    inputs=[],
                    outputs=[
                        helper.make_tensor_value_info(f"if_output{i-1}", TensorProto.FLOAT, [1])
                    ],
                )
                if_node = helper.make_node(
                    "If",
                    inputs=["cond"],
                    outputs=[out_name],
                    then_branch=then_branch,
                    else_branch=else_branch,
                    name=f"depth{i}'s If node",
                )
                inner_if_node = if_node
        graph_nodes = [constant_node, inner_if_node]
        graph = helper.make_graph(
            graph_nodes,
            "input_use_in_if_test",
            inputs=[input_tensor, cond_tensor],
            outputs=[output_tensor],
        )
        model = helper.make_model(graph, producer_name="input_use_in_if_test")

        verify_with_ort_with_inputs(
            model,
            [np.array([3.0], dtype="float32"), np.array([cond])],
            dtype="float32",
            use_vm=True,
            opset=14,
            target=target,
            dev=dev,
        )

    # Confirm that if works with cond as an array or scalar.
    verify_if(num_nested=1, cond=True)
    verify_if(num_nested=1, cond=False)
    verify_if(num_nested=2, cond=True)
    verify_if(num_nested=2, cond=False)


@tvm.testing.parametrize_targets
def test_size(target, dev):
    """test_size"""

    def verify_size(indata):
        node = helper.make_node(
            "Size",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "size_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, [])],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype="int64", use_vm=True, opset=11, target=target, dev=dev
        )

    input_data = np.array([[1, 0], [1, 1]], dtype=np.int64)
    verify_size(input_data)

    input_data = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.int64)
    verify_size(input_data)


@tvm.testing.parametrize_targets
def test_maxunpool(target, dev):
    """test_maxunpool"""

    def verify_maxunpool(data, indices, kernel_shape, strides, output_shape=None, pads=None):
        input_names = ["xT", "xI"]
        input_info = [
            helper.make_tensor_value_info("xT", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("xI", TensorProto.INT64, list(indices.shape)),
        ]
        input_values = [data, indices]
        if output_shape is not None:
            input_names.append("output_shape")
            input_info.append(
                helper.make_tensor_value_info(
                    "output_shape", TensorProto.INT64, list(output_shape.shape)
                )
            )
            input_values.append(output_shape)
        else:
            # Compute expected output shape
            output_shape = np.asarray(([1, 1] + list(strides))) * np.asarray(list(data.shape))
            output_shape += np.asarray(([0, 0] + list(kernel_shape))) - np.asarray(
                ([0, 0] + list(strides))
            )
            if pads is not None:
                output_shape -= np.asarray(
                    [0, 0] + list(np.sum(np.reshape(list(pads), [-1, 2]), axis=-1))
                )
        output_shape = [int(i) for i in output_shape]

        node = helper.make_node(
            "MaxUnpool", inputs=input_names, outputs=["y"], kernel_shape=kernel_shape
        )

        if pads is not None:
            pad_attr = helper.make_attribute("pads", pads)
            node.attribute.append(pad_attr)

        if strides is not None:
            strides_attr = helper.make_attribute("strides", strides)
            node.attribute.append(strides_attr)

        graph = helper.make_graph(
            [node],
            "maxunpool_test",
            inputs=input_info,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(
            model, input_values, use_vm=True, opset=11, target=target, dev=dev
        )

    # Basic test
    x_t = np.array([[[[5, 6], [7, 8]]]], dtype=np.float32)
    x_i = np.array([[[[0, 7], [13, 15]]]], dtype=np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2])
    # Small stride
    verify_maxunpool(x_t, x_i, [2, 2], strides=[1, 1])
    # Big kernel
    verify_maxunpool(x_t, x_i, [3, 3], strides=[2, 2])
    # With output shape
    output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2], output_shape=output_shape)
    # With explicit reverse padding
    pads = np.asarray([1, 1, 1, 1]).astype(np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2], pads=pads)


@tvm.testing.parametrize_targets
def test_softplus(target, dev):
    """test_softplus"""

    def verify_softplus(indata):
        node = helper.make_node(
            "Softplus",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "softplus_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="softplus_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype="float32", use_vm=True, opset=11, target=target, dev=dev
        )

    # Simple case with all signs.
    input_data = np.array([[-1, 0, 1]], dtype=np.float32)
    verify_softplus(input_data)
    # More fancy case.
    input_data = np.random.randn(1, 32, 32, 3).astype("float32")
    verify_softplus(input_data)


@tvm.testing.parametrize_targets
def test_cumsum(target, dev):
    """test_cumsum"""

    def verify_cumsum(indata, axis, exclusive=0, reverse=0, dtype="float32"):
        cumsum_node = onnx.helper.make_node(
            "CumSum",
            inputs=["X", "axis"],
            outputs=["Y"],
        )
        if exclusive != 0:
            exclusive_attr = helper.make_attribute("exclusive", exclusive)
            cumsum_node.attribute.append(exclusive_attr)
        if reverse != 0:
            reverse_attr = helper.make_attribute("reverse", reverse)
            cumsum_node.attribute.append(reverse_attr)
        nodes = [
            make_constant_node("axis", onnx.TensorProto.INT32, [1], [axis]),
            cumsum_node,
        ]
        if dtype == "float32":
            tensor_type = TensorProto.FLOAT
        else:
            tensor_type = TensorProto.INT32
            dtype = "int32"

        graph = helper.make_graph(
            nodes,
            "cumsum_test",
            inputs=[
                helper.make_tensor_value_info("X", tensor_type, list(indata.shape)),
            ],
            outputs=[helper.make_tensor_value_info("Y", tensor_type, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="cumsum_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype=dtype, use_vm=True, opset=11, target=target, dev=dev
        )

    data = (
        np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ]
        )
        .astype(np.float32)
        .reshape((3, 4))
    )

    verify_cumsum(data, 0)
    verify_cumsum(data, 1)
    verify_cumsum(data, 0, 1, 0)
    verify_cumsum(data, 1, 1, 0)
    verify_cumsum(data, 0, 0, 1)
    verify_cumsum(data, 1, 0, 1)
    verify_cumsum(data, 1, 1, 1)
    data = np.random.randn(1, 32, 32, 3).astype("float32")
    verify_cumsum(data, 1)
    data = np.random.randn(1, 32, 32, 3).astype("int32")
    verify_cumsum(data, 0, dtype="int32")
    verify_cumsum(data, 1, dtype="int32")
    verify_cumsum(data, 0, 1, 0, dtype="int32")
    verify_cumsum(data, 1, 1, 0, dtype="int32")
    verify_cumsum(data, 0, 0, 1, dtype="int32")
    verify_cumsum(data, 1, 0, 1, dtype="int32")
    verify_cumsum(data, 1, 1, 1, dtype="int32")


@tvm.testing.parametrize_targets
def test_eyelike(target, dev):
    """test_eyelike"""

    def verify_eyelike(indata, dynamic=False):
        node_list = []
        eyelike_inputs = ["X"]
        input_node_list = [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, list(indata.shape))
        ]
        input_list = [indata]

        if dynamic:
            input_node_list.append(
                helper.make_tensor_value_info("shape", TensorProto.INT64, [len(indata.shape)])
            )
            input_list.append(np.asarray(indata.shape))
            reshape_node = helper.make_node("Reshape", ["X", "shape"], ["X_dyn"])
            eyelike_inputs[0] = "X_dyn"
            node_list += [reshape_node]

        node = helper.make_node(
            "EyeLike",
            inputs=eyelike_inputs,
            outputs=["Y"],
        )
        node_list.append(node)

        graph = helper.make_graph(
            node_list,
            "eyelike_test",
            inputs=input_node_list,
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="eyelike_test")
        verify_with_ort_with_inputs(
            model, input_list, dtype="float32", opset=9, target=target, dev=dev, use_vm=True
        )

    input_data = np.zeros((5, 5), dtype=np.float32)
    verify_eyelike(input_data)
    verify_eyelike(input_data, True)


# The following parametrized tests loads the tests that ONNX ships as
# serialized ONNX files, inputs, and outputs. The goal of this test
# is to ensure the ONNX importer is in line with the ONNX specification.
# To allow these tests to run in CI before all pass, a number of tests
# that are not yet supported are skipped.

onnx_test_node_dir = os.path.join(os.path.dirname(onnx.__file__), "backend", "test", "data", "node")

onnx_test_folders = sorted(
    dirname
    for dirname in os.listdir(onnx_test_node_dir)
    if dirname.startswith("test") and os.path.isdir(os.path.join(onnx_test_node_dir, dirname))
)

unsupported_onnx_tests = [
    "test_batchnorm_epsilon_training_mode",
    "test_batchnorm_example_training_mode",
    "test_bernoulli",
    "test_bernoulli_expanded",
    "test_bernoulli_double",
    "test_bernoulli_double_expanded",
    "test_bernoulli_seed",
    "test_bernoulli_seed_expanded",
    "test_blackmanwindow",
    "test_blackmanwindow_expanded",
    "test_blackmanwindow_symmetric",
    "test_blackmanwindow_symmetric_expanded",
    # the follow cast and castlike cases have lowering issues
    "test_cast_FLOAT_to_STRING",
    "test_cast_STRING_to_FLOAT",
    "test_castlike_FLOAT_to_STRING",
    "test_castlike_FLOAT_to_STRING_expanded",
    "test_castlike_STRING_to_FLOAT",
    "test_castlike_STRING_to_FLOAT_expanded",
    # the following cast and castlike cases segfault
    "test_cast_DOUBLE_to_FLOAT16",
    "test_castlike_DOUBLE_to_FLOAT16",
    "test_castlike_DOUBLE_to_FLOAT16_expanded",
    "test_convtranspose_dilations",
    "test_cumsum_1d",
    "test_cumsum_1d_exclusive",
    "test_cumsum_1d_reverse",
    "test_cumsum_1d_reverse_exclusive",
    "test_cumsum_2d_axis_0",
    "test_cumsum_2d_axis_1",
    "test_cumsum_2d_negative_axis",
    "test_det_2d",
    "test_det_nd",
    "test_dropout_default",
    "test_dropout_default_mask",
    "test_dropout_default_mask_ratio",
    "test_dropout_default_ratio",
    "test_gru_batchwise",
    "test_hammingwindow",
    "test_hammingwindow_expanded",
    "test_hammingwindow_symmetric",
    "test_hammingwindow_symmetric_expanded",
    "test_hannwindow",
    "test_hannwindow_expanded",
    "test_hannwindow_symmetric",
    "test_hannwindow_symmetric_expanded",
    "test_identity_opt",
    "test_identity_sequence",
    "test_if_opt",
    "test_if_seq",
    "test_loop13_seq",
    "test_loop16_seq_none",
    "test_lstm_batchwise",
    "test_maxpool_with_argmax_2d_precomputed_pads",
    "test_maxpool_with_argmax_2d_precomputed_strides",
    "test_maxunpool_export_with_output_shape",
    "test_melweightmatrix",
    # This test fails llvm with a lowering error:
    "test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded",
    "test_qlinearmatmul_3D",
    "test_range_float_type_positive_delta_expanded",
    "test_range_int32_type_negative_delta_expanded",
    "test_reduce_sum_do_not_keepdims_example",
    "test_reduce_sum_do_not_keepdims_random",
    "test_reduce_sum_keepdims_example",
    "test_reduce_sum_keepdims_random",
    "test_reduce_sum_negative_axes_keepdims_example",
    "test_reduce_sum_negative_axes_keepdims_random",
    "test_roialign_aligned_true",
    "test_sequence_insert_at_back",
    "test_sequence_insert_at_front",
    "test_sequence_map_add_1_sequence_1_tensor",
    "test_sequence_map_add_1_sequence_1_tensor_expanded",
    "test_sequence_map_add_2_sequences",
    "test_sequence_map_add_2_sequences_expanded",
    "test_sequence_map_extract_shapes",
    "test_sequence_map_extract_shapes_expanded",
    "test_sequence_map_identity_1_sequence",
    "test_sequence_map_identity_1_sequence_1_tensor",
    "test_sequence_map_identity_1_sequence_1_tensor_expanded",
    "test_sequence_map_identity_1_sequence_expanded",
    "test_sequence_map_identity_2_sequences",
    "test_sequence_map_identity_2_sequences_expanded",
    "test_simple_rnn_batchwise",
    "test_simple_rnn_defaults",
    "test_simple_rnn_with_initial_bias",
    "test_split_variable_parts_1d",
    "test_split_variable_parts_2d",
    "test_split_variable_parts_default_axis",
    "test_split_zero_size_splits",
    "test_stft",
    "test_stft_with_window",
    "test_strnormalizer_export_monday_casesensintive_lower",
    "test_strnormalizer_export_monday_casesensintive_nochangecase",
    "test_strnormalizer_export_monday_casesensintive_upper",
    "test_strnormalizer_export_monday_empty_output",
    "test_strnormalizer_export_monday_insensintive_upper_twodim",
    "test_strnormalizer_nostopwords_nochangecase",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip0",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_batch_uniandbigrams_skip5",
    "test_tfidfvectorizer_tf_only_bigrams_skip0",
    "test_tfidfvectorizer_tf_onlybigrams_levelempty",
    "test_tfidfvectorizer_tf_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_uniandbigrams_skip5",
    "test_training_dropout",
    "test_training_dropout_default",
    "test_training_dropout_default_mask",
    "test_training_dropout_mask",
    "test_training_dropout_zero_ratio",
    "test_training_dropout_zero_ratio_mask",
    "test_tril_zero",
    "test_triu_zero",
    "test_unique_sorted_with_axis",
    "test_unique_sorted_with_axis_3d",
    "test_unique_sorted_with_negative_axis",
    "test_upsample_nearest",
    "test_upsample_nearest_default",
]


target_skips = {
    "cuda": [
        "test_range_float_type_positive_delta_expanded",
        "test_range_int32_type_positive_delta_expanded",
        "test_mod_mixed_sign_float16",
        "test_qlinearconv",
        "test_qlinearmatmul",
        "test_resize_upsample_sizes_nearest",
    ]
}


def _load_proto(proto_filename, target_list, model_type_proto):
    with open(proto_filename, "rb") as fin:
        protobuf_content = fin.read()
        if model_type_proto.HasField("sequence_type"):
            sequence = onnx.SequenceProto()
            sequence.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_list(sequence))
        elif model_type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_array(tensor))
        elif model_type_proto.HasField("optional_type"):
            optional = onnx.OptionalProto()
            optional.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_optional(optional))
        else:
            raise ValueError(
                "Loading proto of that specific type (Map/Sparse Tensor) is currently not supported"
            )


def is_ort_version_lower_than(ver):
    import onnxruntime as ort

    v11, v12, v13 = tuple(int(v) for v in ort.__version__.split("."))
    v21, v22, v23 = tuple(int(v) for v in ver.split("."))

    return (v11 < v21) or (v11 == v21 and v12 < v22) or ((v11, v12) == (v21, v22) and v13 < v23)


@pytest.mark.parametrize("onnx_test", onnx_test_folders)
@tvm.testing.parametrize_targets
def test_onnx_nodes(target, dev, onnx_test):
    """test_onnx_nodes"""
    if platform.machine() == "aarch64" and onnx_test == "test_resize_upsample_sizes_nearest":
        pytest.skip("Currently failing on AArch64")

    target_kind = tvm.target.Target(target).kind.name

    if onnx_test in unsupported_onnx_tests:
        pytest.skip(f"Onnx test '{onnx_test}' not yet supported by TVM")

    target_specific_skips = target_skips.get(target_kind, [])
    if onnx_test in target_specific_skips:
        pytest.skip(f"Onnx test '{onnx_test}' not yet supported by TVM on {target_kind} targets")

    if is_ort_version_lower_than("1.13.1") and onnx_test == "test_convtranspose_autopad_same":
        pytest.skip(
            f"Onnx test '{onnx_test}' expected to fail for onnxruntime version lower than 1.13.1 "
            "due to different interpretation of auto_pad parameters SAME_UPPER and SAME_LOWER."
        )

    test_dir = os.path.join(onnx_test_node_dir, onnx_test)

    atol = 1e-5
    rtol = 1e-5
    if "roialign" in test_dir:
        # for some reason the ONNX test crops the
        # roialign results to 4 decimal places
        atol = 1e-4

    if "to_BFLOAT16" in test_dir:
        # the tolerance here is for the comparison in uint16 space, but is not as significant
        # of a delta in bfloat16 space because it's representing the mantissa being off by 1
        atol = 1

    if "_sce_" in test_dir:
        # complicated loss functions like SoftmaxCrossEntropy can have minor variations
        # in accuracy depending on implementation
        atol = 1e-4

    if "bicubic" in test_dir:
        # satisfies onnx precision for bicubic interpolation
        atol = 1e-4

    if "dft" in test_dir:
        atol = 1e-3

    model = onnx.load(os.path.join(test_dir, "model.onnx"))
    for test_data_dir in glob.glob(os.path.join(test_dir, "test_data_set*")):
        inputs = []
        n_inputs = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
        for i in range(n_inputs):
            input_file = os.path.join(test_data_dir, f"input_{i}.pb")
            _load_proto(input_file, inputs, model.graph.input[i].type)

        outputs = []
        n_outputs = len(glob.glob(os.path.join(test_data_dir, "output_*.pb")))
        for i in range(n_outputs):
            output_file = os.path.join(test_data_dir, f"output_{i}.pb")
            _load_proto(output_file, outputs, model.graph.output[i].type)

    tvm_val = get_tinygrad_output_with_vm(model, inputs, target, dev)
    if len(outputs) == 1:
        tvm.testing.assert_allclose(outputs[0], tvm_val, rtol=rtol, atol=atol)
    else:
        for output, val in zip(outputs, tvm_val):
            tvm.testing.assert_allclose(output, val, rtol=rtol, atol=atol)


def test_wrong_input():
    """test_wrong_input"""
    node = helper.make_node(
        "Softplus",
        inputs=["X"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [node],
        "softplus_test",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list([5]))],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list([5]))],
    )
    model = helper.make_model(graph, producer_name="softplus_test")

    # Check that the graph can import correctly with proper shape definitions.
    correct_shape_dict = {"X": [5]}
    relay.frontend.from_onnx(model, shape=correct_shape_dict)

    # Check that an assertion is triggered when an input not in the graph is provided.
    wrong_shape_dict = {"Z": [5]}
    with pytest.raises(AssertionError):
        relay.frontend.from_onnx(model, shape=wrong_shape_dict)


@pytest.mark.skip(reason="unsupported op numel")
@tvm.testing.parametrize_targets
def test_aten(target, dev):
    """test_aten"""
    torch.set_grad_enabled(False)

    def _convert_to_onnx(model, inputs):
        file_name = "aten_model.onnx"
        torch.onnx.export(
            model,
            inputs,
            file_name,
            export_params=True,
            verbose=False,
            opset_version=10,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
        )
        onnx_model = onnx.load(file_name)
        return onnx_model

    def verify_embedding_bag(num_embedding, embedding_dim, data_shape, num_bags=None):
        dummy_data = torch.randint(0, num_embedding - 1, data_shape)
        tvm_inputs = [dummy_data.numpy()]
        model = torch.nn.EmbeddingBag(num_embedding, embedding_dim)
        onnx_model = _convert_to_onnx(model, dummy_data)
        torch_out = model(dummy_data)
        tvm_out = get_tinygrad_output_with_vm(
            onnx_model,
            tvm_inputs,
            freeze_params=True,
            target=target,
            dev=dev,
        )
        tvm.testing.assert_allclose(torch_out.numpy(), tvm_out, atol=5e-7)

    verify_embedding_bag(10, 3, [2, 10])
    verify_embedding_bag(32, 2, [3, 3])


@tvm.testing.parametrize_targets
def test_index_put(target, dev):
    """test_index_put"""

    class IndexPutModel(torch.nn.Module):
        def __init__(self, indices, values, accumulate):
            super().__init__()
            self.indices = indices
            self.values = values
            self.accumulate = accumulate

        def forward(self, x):
            return x.index_put(self.indices, self.values, self.accumulate)

    def _convert_to_onnx(model, dummy_data):
        file_name = "aten_model.onnx"
        torch.onnx.export(
            model,
            dummy_data,
            file_name,
            export_params=True,
            verbose=False,
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        onnx_model = onnx.load(file_name)
        return onnx_model

    def verify_index_put(data_shape, indices, accumulate):
        dummy_data = torch.ones(data_shape)
        tvm_inputs = [dummy_data.numpy()]
        values = torch.rand(indices[0].size())
        model = IndexPutModel(indices, values, accumulate)
        onnx_model = _convert_to_onnx(model, dummy_data)
        torch_out = model(dummy_data)

        tvm_out = get_tinygrad_output_with_vm(onnx_model, tvm_inputs, target, dev, freeze_params=True)
        tvm.testing.assert_allclose(torch_out.numpy(), tvm_out)

    shape = (3, 5)
    xidx = torch.tensor([0, 1, 2, 2])
    yidx = torch.tensor([0, 1, 3, 4])
    verify_index_put(shape, [xidx, yidx], True)

    shape = (3, 5, 3)
    xidx = torch.tensor([0, 1, 2, 2, 0])
    yidx = torch.tensor([0, 1, 3, 4, 0])
    zidx = torch.tensor([0, 1, 1, 2, 0])
    verify_index_put(shape, [xidx, yidx, zidx], False)

    def verify_index_put_slice(data_shape, value_shape, accumulate):
        dummy_data = torch.ones(data_shape)
        tvm_inputs = [dummy_data.numpy()]
        indices = []
        index_shape = [1] * len(value_shape)
        index_shape[0] = -1
        for _, v_shape in enumerate(value_shape):
            indices.append(torch.arange(0, v_shape).reshape(tuple(index_shape)))
            index_shape.pop()
        values = torch.rand(value_shape)

        model = IndexPutModel(indices, values, accumulate)
        onnx_model = _convert_to_onnx(model, dummy_data)
        torch_out = model(dummy_data)

        tvm_out = get_tinygrad_output_with_vm(onnx_model, tvm_inputs, target, dev, freeze_params=True)
        tvm.testing.assert_allclose(torch_out.numpy(), tvm_out)

    verify_index_put_slice((3, 3), (2, 2), False)
    verify_index_put_slice((2, 3, 4), (1, 2, 3), True)
    verify_index_put_slice((2, 3, 4, 5), (1, 2, 3, 1), False)


@tvm.testing.parametrize_targets
def test_reverse_sequence(target, dev):
    """test_reverse_sequence"""

    def verify_reverse_sequence(x, sequence_lens, batch_axis, time_axis):
        node = onnx.helper.make_node(
            "ReverseSequence",
            inputs=["x", "sequence_lens"],
            outputs=["y"],
            time_axis=time_axis,
            batch_axis=batch_axis,
        )

        graph = helper.make_graph(
            [node],
            "reverse_sequence_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
                helper.make_tensor_value_info(
                    "sequence_lens", TensorProto.INT64, list(sequence_lens.shape)
                ),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name="reverse_sequence_test")
        verify_with_ort_with_inputs(model, [x, sequence_lens], [x.shape], target=target, dev=dev)

    x = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        dtype=np.float32,
    )
    sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)
    verify_reverse_sequence(x, sequence_lens, 0, 1)

    sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)
    verify_reverse_sequence(x, sequence_lens, 1, 0)


@pytest.mark.parametrize("op_name", ["Gelu", "FastGelu"], scope="session")
@pytest.mark.parametrize("data_type", ["float16", "float32"], scope="session")
@tvm.testing.parametrize_targets
def test_gelu(target, dev, data_type, op_name):
    """test_gelu"""
    dtype = np.dtype(data_type)
    tensor_type = mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    absolute_tolerance = 1e-3 if data_type == "float16" else 1e-5

    def verify_gelu(x):
        node = onnx.helper.make_node(
            op_name,
            inputs=["x"],
            outputs=["y"],
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            f"{op_name}_test",
            inputs=[helper.make_tensor_value_info("x", tensor_type, list(x.shape))],
            outputs=[helper.make_tensor_value_info("y", tensor_type, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name=f"{op_name}_test")
        verify_with_ort_with_inputs(
            model, [x], [x.shape], atol=absolute_tolerance, dtype=data_type, target=target, dev=dev
        )

    x = np.array([-1.0, 0, 1.0, 100.0, -100.0, 1000.0, -1000.0], dtype=dtype)
    verify_gelu(x)
    x = np.array([[1, 2], [3, 4]], dtype=dtype)
    verify_gelu(x)


@pytest.mark.parametrize("op_name", ["BiasGelu", "FastGelu"], scope="session")
@pytest.mark.parametrize("data_type", ["float16", "float32"], scope="session")
@tvm.testing.parametrize_targets
def test_biasgelu(target, dev, data_type, op_name):
    """test_biasgelu"""
    dtype = np.dtype(data_type)
    tensor_type = mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    absolute_tolerance = 1e-2 if data_type == "float16" else 1e-5

    def verify_biasgelu(x, bias):
        node = onnx.helper.make_node(
            op_name,
            inputs=["x", "bias"],
            outputs=["y"],
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            f"{op_name}_test",
            inputs=[
                helper.make_tensor_value_info("x", tensor_type, list(x.shape)),
                helper.make_tensor_value_info("bias", tensor_type, list(bias.shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", tensor_type, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name=f"{op_name}_test")
        verify_with_ort_with_inputs(
            model,
            [x, bias],
            [x.shape],
            atol=absolute_tolerance,
            dtype=data_type,
            target=target,
            dev=dev,
        )

    x = np.array([-1.0, 0, 1.0, 100.0, -100.0, 1000.0, -1000.0], dtype=dtype)
    bias = np.repeat(2.0, 7).astype(dtype)
    verify_biasgelu(x, bias)

    x = np.array([[1, 2], [3, 4]], dtype=dtype)
    bias = np.array([0.3, 4.0], dtype=dtype)
    verify_biasgelu(x, bias)


@tvm.testing.parametrize_targets
def test_embedlayernormalization(target, dev):
    """test_embedlayernormalization"""

    def verify_embedlayernormalization(
        input_ids,
        segment_ids,
        word_embedding,
        position_embedding,
        segment_embedding,
        gamma,
        beta,
    ):
        node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                "input_ids",
                "" if segment_ids is None else "segment_ids",
                "word_embedding",
                "position_embedding",
                "" if segment_embedding is None else "segment_embedding",
                "gamma",
                "beta",
            ],
            outputs=["output", "mask_index"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        segment_ids_shape = [] if segment_ids is None else segment_ids.shape
        segment_embedding_shape = [] if segment_embedding is None else segment_embedding.shape

        graph = helper.make_graph(
            [node],
            "embedlayernormalization_test",
            inputs=[
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT32, list(input_ids.shape)
                ),
                helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape),
                helper.make_tensor_value_info(
                    "word_embedding", TensorProto.FLOAT, list(word_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "position_embedding", TensorProto.FLOAT, list(position_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "segment_embedding", TensorProto.FLOAT, segment_embedding_shape
                ),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, list((batch_size, sequence_length, hidden_size))
                ),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size]),
            ],
        )

        model = helper.make_model(graph, producer_name="embedlayernormalization_test")

        # TODO(@anwang2009): onnxruntime v1.9.0 requires empty list for optional argument,
        # but v1.10.0+ requires None instead.
        verify_with_ort_with_inputs(
            model,
            [
                input_ids,
                np.empty(0, dtype="int32") if segment_ids is None else segment_ids,
                word_embedding,
                position_embedding,
                np.empty(0, dtype="float32") if segment_embedding is None else segment_embedding,
                gamma,
                beta,
            ],
            [
                (batch_size, sequence_length, hidden_size),
                batch_size,
            ],
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
        )

    hidden_size = 384
    batch_size = 4
    sequence_length = 3
    vocab_size = 5

    input_ids = np.full((batch_size, sequence_length), 3).astype("int32")
    segment_ids = np.zeros((batch_size, sequence_length)).astype("int32")
    word_embedding = np.full((vocab_size, hidden_size), 1).astype("float32")
    position_embedding = np.full((sequence_length, hidden_size), 2).astype("float32")
    segment_embedding = np.full((vocab_size, hidden_size), 3).astype("float32")

    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype("float32")
    beta = np.random.randn(hidden_size).astype("float32") * 0.1

    verify_embedlayernormalization(
        input_ids, segment_ids, word_embedding, position_embedding, segment_embedding, gamma, beta
    )

    # Test with undefined segment embedding
    verify_embedlayernormalization(
        input_ids, None, word_embedding, position_embedding, None, gamma, beta
    )


@tvm.testing.parametrize_targets
def test_attention(target, dev):
    """test_attention"""

    def verify_attention(_unidirectional, _input, _weight, _bias, _mask_index=None, _past=None):
        input_names = ["input", "weight", "bias"]
        if _mask_index is not None:
            input_names.append("mask_index")
        if _past is not None:
            input_names.append("past")

        node = onnx.helper.make_node(
            "Attention",
            inputs=input_names,
            outputs=["output", "present"],
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=_unidirectional,
        )

        past_shape = (2, batch_size, num_heads, past_sequence_length, head_size)
        present_output_shape = (2, batch_size, num_heads, sequence_length, head_size)

        inputs_info = [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, list(_input.shape)),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, list(_weight.shape)),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(_bias.shape)),
        ]
        if _mask_index is not None:
            inputs_info.append(
                helper.make_tensor_value_info(
                    "mask_index", TensorProto.INT32, list(_mask_index.shape)
                ),
            )
        if _past is not None:
            inputs_info.append(
                helper.make_tensor_value_info("past", TensorProto.FLOAT, list(past_shape))
            )

        graph = helper.make_graph(
            [node],
            "attention_test",
            inputs=inputs_info,
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(_input.shape)),
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, list(present_output_shape)
                ),
            ],
        )

        model = helper.make_model(graph, producer_name="attention_test")

        inputs = [_input, _weight, _bias]
        if _mask_index is not None:
            inputs.append(_mask_index)
        if _past is not None:
            inputs.append(_past)

        # "present" output should be nullptr when the "past" input isn't included,
        # but ort requires an output shape to be specified?
        verify_with_ort_with_inputs(
            model,
            inputs,
            [_input.shape, present_output_shape],
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
        )

    batch_size = 11
    num_heads = 13
    head_size = 37
    sequence_length = 7
    input_hidden_size = 147
    weight_hidden_size = num_heads * head_size
    past_sequence_length = 17

    total_sequence_length = past_sequence_length + sequence_length

    # Required inputs
    input_array = np.random.normal(size=(batch_size, sequence_length, input_hidden_size)).astype(
        "float32"
    )
    weight = (
        np.random.normal(size=(input_hidden_size, 3 * weight_hidden_size)).astype("float32") * 0.1
    )
    bias = np.random.randn(3 * weight_hidden_size).astype("float32")

    # Optional inputs
    past = np.random.random((2, batch_size, num_heads, past_sequence_length, head_size)).astype(
        "float32"
    )

    for unidirectional in [0, 1]:
        for have_past in [False, True]:
            if not have_past:
                mask_index = np.random.randint(0, 2, (batch_size, sequence_length)).astype("int32")
                verify_attention(unidirectional, input_array, weight, bias, mask_index)
            else:
                mask_index = np.random.randint(0, 2, (batch_size, total_sequence_length)).astype(
                    "int32"
                )
                verify_attention(unidirectional, input_array, weight, bias, mask_index, past)


@tvm.testing.parametrize_targets
def test_qattention(target, dev):
    """test_qattention"""

    def verify_attention(
        _unidirectional,
        _input,
        _weight,
        _bias,
        _input_scale,
        _weight_scale,
        _mask_index=None,
        _input_zero_point=None,
        _weight_zero_point=None,
        _past=None,
    ):
        input_names = ["input", "weight", "bias", "input_scale", "weight_scale"]
        if _mask_index is not None:
            input_names.append("mask_index")
        if _input_zero_point is not None:
            input_names.append("input_zero_point")
        if _weight_zero_point is not None:
            input_names.append("weight_zero_point")
        if _past is not None:
            input_names.append("past")

        node = onnx.helper.make_node(
            "QAttention",
            inputs=input_names,
            outputs=["output", "present"],
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=_unidirectional,
        )

        past_shape = (2, batch_size, num_heads, past_sequence_length, head_size)
        present_output_shape = (
            2,
            batch_size,
            num_heads,
            past_sequence_length + sequence_length,
            head_size,
        )

        inputs_info = [
            helper.make_tensor_value_info("input", TensorProto.UINT8, list(_input.shape)),
            helper.make_tensor_value_info("weight", TensorProto.UINT8, list(_weight.shape)),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(_bias.shape)),
            helper.make_tensor_value_info("input_scale", TensorProto.FLOAT, ()),
            helper.make_tensor_value_info("weight_scale", TensorProto.FLOAT, ()),
        ]
        if _mask_index is not None:
            inputs_info.append(
                helper.make_tensor_value_info(
                    "mask_index", TensorProto.INT32, list(_mask_index.shape)
                )
            )
        if _input_zero_point is not None:
            inputs_info.append(
                helper.make_tensor_value_info("input_zero_point", TensorProto.UINT8, ())
            )
        if _weight_zero_point is not None:
            inputs_info.append(
                helper.make_tensor_value_info("weight_zero_point", TensorProto.UINT8, ())
            )
        if _past is not None:
            inputs_info.append(
                helper.make_tensor_value_info("past", TensorProto.FLOAT, list(past_shape))
            )

        graph = helper.make_graph(
            [node],
            "qattention_test",
            inputs=inputs_info,
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(_input.shape)),
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, list(present_output_shape)
                ),
            ],
        )

        model = helper.make_model(graph, producer_name="qattention_test")

        inputs = [_input, _weight, _bias, _input_scale, _weight_scale]
        if _mask_index is not None:
            inputs.append(_mask_index)
        if _input_zero_point is not None:
            inputs.append(_input_zero_point)
        if _weight_zero_point is not None:
            inputs.append(_weight_zero_point)
        if _past is not None:
            inputs.append(_past)

        verify_with_ort_with_inputs(
            model,
            inputs,
            [_input.shape, present_output_shape],
            target=target,
            dev=dev,
            rtol=1e-3,
            atol=1e-3,
        )

    batch_size = 11
    num_heads = 13
    head_size = 37
    sequence_length = 7
    input_hidden_size = 147
    weight_hidden_size = num_heads * head_size
    past_sequence_length = 17

    total_sequence_length = past_sequence_length + sequence_length

    # Required inputs
    input_array = np.random.randint(
        0, 255, (batch_size, sequence_length, input_hidden_size)
    ).astype("uint8")
    weight = np.random.randint(0, 255, (input_hidden_size, 3 * weight_hidden_size)).astype("uint8")
    bias = np.random.randn(3 * weight_hidden_size).astype("float32")
    input_scale = np.random.random(1).astype("float32")
    weight_scale = np.random.random(1).astype("float32")

    # Optional inputs
    input_zero_point = np.random.randint(0, 255, 1).astype("uint8")
    weight_zero_point = np.random.randint(0, 255, 1).astype("uint8")
    past = np.random.random((2, batch_size, num_heads, past_sequence_length, head_size)).astype(
        "float32"
    )

    for unidirectional in [0, 1]:
        for have_past in [False, True]:
            if not have_past:
                mask_index = np.random.randint(0, 2, (batch_size, sequence_length)).astype("int32")

                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                )
                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                    input_zero_point,
                )
                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                    input_zero_point,
                    weight_zero_point,
                )
            else:
                mask_index = np.random.randint(0, 2, (batch_size, total_sequence_length)).astype(
                    "int32"
                )

                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                    input_zero_point,
                    weight_zero_point,
                    past,
                )


@tvm.testing.parametrize_targets
def test_skiplayernormalization(target, dev):
    """test_skiplayernormalization"""

    def verify_skiplayernormalization(input_, skip, gamma, beta, bias):
        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta", "bias"],
            outputs=["output"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        graph = helper.make_graph(
            [node],
            "skiplayernormalization_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_.shape)),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, list(skip.shape)),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(bias.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(input_.shape)),
            ],
        )

        model = helper.make_model(graph, producer_name="skiplayernormalization_test")
        verify_with_ort_with_inputs(
            model, [input_, skip, gamma, beta, bias], [input_.shape], target=target, dev=dev
        )

    hidden_size = 384
    batch_size = 4
    sequence_length = 4

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    skip = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype(dtype)
    beta = np.random.randn(hidden_size).astype(dtype) * 0.1
    bias = np.random.randn(hidden_size).astype(dtype)

    verify_skiplayernormalization(input_array, skip, gamma, beta, bias)


@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_qgemm(target, dev):
    """test_qgemm"""

    def verify_qgemm(
        a_shape,
        b_shape,
        y_shape,
        C=False,
        y_zp=False,
        b_per_tensor_quantization=False,
        alpha=1.0,
        transA=0,
        transB=1,
    ):
        a_array = np.random.randint(low=0, high=255, size=a_shape).astype("uint8")
        b_array = np.random.uniform(low=0, high=255, size=b_shape).astype("uint8")

        input_nodes = [
            helper.make_tensor_value_info("a", TensorProto.UINT8, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.UINT8, list(b_shape)),
        ]

        initializer = [
            helper.make_tensor("a_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor("a_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
        ]

        input_names = [
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
        ]
        input_values = [a_array, b_array]

        if b_per_tensor_quantization:
            initializer.append(
                helper.make_tensor("b_scale", TensorProto.FLOAT, (), [np.random.rand()])
            )
            initializer.append(
                helper.make_tensor(
                    "b_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]
                )
            )
        else:  # per_colume_quantization
            shape_value = b_shape[0] if transB else b_shape[1]
            b_scale_array = np.random.random(shape_value).astype("float32")
            w_zero_point_array = np.random.randint(0, 255, size=shape_value).astype("uint8")
            initializer.append(
                helper.make_tensor(
                    "b_scale", TensorProto.FLOAT, list(b_scale_array.shape), b_scale_array
                )
            )
            initializer.append(
                helper.make_tensor(
                    "b_zero_point",
                    TensorProto.UINT8,
                    list(w_zero_point_array.shape),
                    w_zero_point_array,
                )
            )

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(y_shape))

        if C is True:
            C_shape = (b_shape[0] if transB else b_shape[1],)
            C_array = np.random.randint(low=0, high=65536, size=C_shape).astype("int32")
            input_nodes.append(helper.make_tensor_value_info("C", TensorProto.INT32, list(C_shape)))
            input_names.append("C")
            input_values.append(C_array)

        if y_zp is True:
            input_names.append("y_scale")
            initializer.append(
                helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()])
            )

            input_names.append("y_zero_point")
            initializer.append(
                helper.make_tensor(
                    "y_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]
                )
            )

            output_tensor = helper.make_tensor_value_info(
                "output", TensorProto.UINT8, list(y_shape)
            )

        kwargs = {}
        kwargs["alpha"] = alpha
        kwargs["transA"] = transA
        kwargs["transB"] = transB

        node = helper.make_node(
            "QGemm",
            inputs=input_names,
            outputs=["output"],
            domain="com.microsoft",
            # Default values for other attributes:
            **kwargs,
        )

        graph = helper.make_graph(
            [node],
            "QGemm",
            inputs=input_nodes,
            outputs=[output_tensor],
            initializer=initializer,
        )
        model = helper.make_model(
            graph,
            producer_name="QGemm",
            opset_imports=[
                onnx.helper.make_opsetid("com.microsoft", 1),
            ],
        )

        verify_with_ort_with_inputs(model, input_values, target=target, dev=dev)

    # B per tensor quantization
    verify_qgemm(
        (20, 30),
        (50, 30),
        (20, 50),
        True,
        True,
        True,
    )

    # B per column  quantization
    verify_qgemm(
        (20, 30),
        (50, 30),
        (20, 50),
        True,
        True,
        False,
    )

    # test alpha
    verify_qgemm(
        (20, 30),
        (50, 30),
        (20, 50),
        True,
        True,
        True,
        0.5,
    )

    # test transpose A
    verify_qgemm(
        (20, 50),
        (20, 80),
        (50, 80),
        True,
        True,
        True,
        0.5,
        1,
        0,
    )


@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_qlinearconv(target, dev):
    """test_qlinearconv"""

    def verify_qlinearconv(
        x_shape,
        w_shape,
        y_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        auto_pad="NOTSET",
        bias=False,
        per_channel_quantization=False,
    ):

        x_array = np.random.randint(low=0, high=255, size=x_shape).astype("uint8")
        w_array = np.random.uniform(low=0, high=255, size=w_shape).astype("uint8")

        initializer = [
            helper.make_tensor("x_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor("x_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
            helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor("y_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
        ]

        input_nodes = [
            helper.make_tensor_value_info("x", TensorProto.UINT8, list(x_shape)),
            helper.make_tensor_value_info("w", TensorProto.UINT8, list(w_shape)),
        ]
        input_names = [
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ]
        input_values = [x_array, w_array]

        if per_channel_quantization:
            w_scale_array = np.random.random(w_shape[0]).astype("float32")
            w_zero_point_array = np.random.randint(0, 255, size=w_shape[0]).astype("uint8")

            initializer.append(
                helper.make_tensor("w_scale", TensorProto.FLOAT, [w_shape[0]], w_scale_array)
            )
            initializer.append(
                helper.make_tensor(
                    "w_zero_point", TensorProto.UINT8, [w_shape[0]], w_zero_point_array
                )
            )
        else:
            initializer.append(
                helper.make_tensor("w_scale", TensorProto.FLOAT, (), [np.random.rand()])
            )
            initializer.append(
                helper.make_tensor(
                    "w_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]
                )
            )

        if bias is True:
            b_shape = w_shape[0:1]
            b_array = np.random.randint(low=0, high=65536, size=b_shape).astype("int32")
            input_nodes.append(helper.make_tensor_value_info("B", TensorProto.INT32, list(b_shape)))
            input_names.append("B")
            input_values.append(b_array)

        if padding is None:
            ## autopadding with unset default attributes
            kwargs = {}
            if not all(list(s == 1 for s in strides)):
                kwargs["strides"] = strides
            if not all(list(d == 1 for d in dilations)):
                kwargs["dilations"] = dilations

            node = helper.make_node(
                "QLinearConv",
                inputs=input_names,
                outputs=["y"],
                # Default values for other attributes:
                auto_pad=auto_pad,
                **kwargs,
            )
        else:
            node = helper.make_node(
                "QLinearConv",
                inputs=input_names,
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                # groups=1
                pads=padding,
            )

        graph = helper.make_graph(
            [node],
            "conv_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("y", TensorProto.UINT8, list(y_shape))],
            initializer=initializer,
        )
        model = helper.make_model(graph, producer_name="qlinearconv_test")
        # opt_level=1 will cause error
        verify_with_ort_with_inputs(model, input_values, opt_level=2, target=target, dev=dev)

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    # only support QLinearConv2d because only support qnn.conv2d
    dims = 2

    # Convolution with padding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )

    # Convolution with bias
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        bias=True,
    )

    # Convolution with asymmetric padding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(4, dims),
        repeat(0, dims) + repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution without padding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        2 * repeat(0, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution with autopadding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with valid autopadding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="VALID",
    )
    # Convolution with non uniform stride
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(2, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with dilation
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(2, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(2, dims),
    )
    # Convolution with per channel quantization
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        per_channel_quantization=True,
    )


# TODO(vvchernov): fix problem with quantization on cuda
@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_qlinearmatmul(target, dev):
    """test_qlinearmatmul"""

    def verify_qlinearmatmul(
        x_shape,
        w_shape,
        y_shape,
        x_dtype="uint8",
        w_dtype="uint8",
    ):
        def get_randint_numpy_scalar(dtype="uint8"):
            if dtype == "uint8":
                return np.random.randint(0, 255)
            else:  # "int8"
                return np.random.randint(-128, 127)

        if x_dtype == "uint8":
            x_array = np.random.randint(low=0, high=255, size=x_shape).astype("uint8")
        else:  # "int8"
            x_array = np.random.randint(low=-128, high=127, size=x_shape).astype("int8")
        if w_dtype == "uint8":
            w_array = np.random.uniform(low=0, high=255, size=w_shape).astype("uint8")
        else:  # "int8"
            w_array = np.random.uniform(low=-128, high=127, size=w_shape).astype("int8")

        x_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(x_dtype)]
        w_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(w_dtype)]

        y_dtype = "int8"
        if x_dtype == "uint8" and w_dtype == "uint8":
            y_dtype = "uint8"
        y_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(y_dtype)]

        initializer = [
            helper.make_tensor("x_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            # TODO: 0 value for int8?
            helper.make_tensor(
                "x_zero_point", x_proto_type, (), [get_randint_numpy_scalar(x_dtype)]
            ),
            helper.make_tensor("w_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            # TODO: 0 value for int8?
            helper.make_tensor(
                "w_zero_point", w_proto_type, (), [get_randint_numpy_scalar(w_dtype)]
            ),
            helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor(
                "y_zero_point", y_proto_type, (), [get_randint_numpy_scalar(y_dtype)]
            ),
        ]

        input_nodes = [
            helper.make_tensor_value_info("x", x_proto_type, list(x_shape)),
            helper.make_tensor_value_info("w", w_proto_type, list(w_shape)),
        ]
        input_names = [
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ]
        input_values = [x_array, w_array]

        node = helper.make_node(
            "QLinearMatMul",
            inputs=input_names,
            outputs=["y"],
        )

        y_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("int8")]
        if x_dtype == "uint8" and w_dtype == "uint8":
            y_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("uint8")]

        graph = helper.make_graph(
            [node],
            "qmatmul_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("y", y_proto_type, list(y_shape))],
            initializer=initializer,
        )
        model = helper.make_model(graph, producer_name="qlinearmatmul_test")
        # opt_level=1 will cause error
        verify_with_ort_with_inputs(model, input_values, opt_level=2, target=target, dev=dev)

    # Default matmul both ranks = 2 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 3), (3, 2), (2, 2))

    # Default matmul both ranks = 2 (x_dtype = "int8", w_dtype = "int8")
    verify_qlinearmatmul((2, 3), (3, 2), (2, 2), "int8", "int8")

    # TODO(vvchernov): problems on ONNX Runtime side and type check (onnx.py:L4763) on TVM side
    # Default matmul both ranks = 2 (x_dtype = "uint8", w_dtype = "int8")
    # verify_qlinearmatmul((2, 3), (3, 2), (2, 2), "uint8", "int8")

    # TODO(vvchernov): problems on ONNX Runtime side and type check (onnx.py:L4763) on TVM side
    # Default matmul both ranks = 2 (x_dtype = "int8", w_dtype = "uint8")
    # verify_qlinearmatmul((2, 3), (3, 2), (2, 2), "int8", "uint8")

    # Reduced matmul: x_ranks = 1, w_rank = 2 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((3,), (3, 2), (2,))

    # Special case matmul: x_ranks = 3, w_rank = 2 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 3, 4), (4, 3), (2, 3, 3))

    # GPT2-style matmul both ranks = 4 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 4, 3, 3), (2, 4, 3, 3), (2, 4, 3, 3))

    # Asymetric matmul: x_ranks = 4, w_rank = 3 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 4, 3, 3), (4, 3, 3), (2, 4, 3, 3))

    # Asymetric matmul: x_ranks = 2, w_rank = 3 (x_dtype = "uint8", w_dtype = "uint8")
    # verify_qlinearmatmul((3, 3), (4, 3, 3), (4, 3, 3))


@tvm.testing.parametrize_targets
def test_qlinearconcat(target, dev):
    """test_qlinearconcat"""

    def verify_qlinearconcat(shapes, out_shape, axis=None):
        input_names = []
        input_values = []
        input_nodes = []
        for i, shape in enumerate(shapes):
            tensor_name = chr(ord("a") + i)
            node = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, list(shape))

            input_names.append(tensor_name)
            input_values.append(np.random.random(shape).astype("float32"))
            input_nodes.append(node)

        node = helper.make_node("Concat", input_names, ["C"])
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            node.attribute.append(axis_attr)
        graph = helper.make_graph(
            [node],
            "qlinearconcat_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("C", TensorProto.FLOAT, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="qlinearconcat_test")
        quantize_and_verify_with_ort(model, input_names, shapes, target, dev)

    verify_qlinearconcat([[2, 1], [2, 1]], [4, 1], 0)
    verify_qlinearconcat([[2, 1], [2, 1]], [2, 2], 1)
    verify_qlinearconcat([[1, 2], [2, 2], [3, 2]], [6, 2], 0)


@tvm.testing.parametrize_targets
def test_qlinearadd(target, dev):
    """test_qlinearadd"""

    def verify_qlinearadd(a_shape, b_shape, c_shape):

        _ = np.random.random(a_shape).astype("float32")
        _ = np.random.random(b_shape).astype("float32")

        input_nodes = [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ]
        input_names = [
            "a",
            "b",
        ]

        node = helper.make_node("Add", ["a", "b"], ["C"])
        graph = helper.make_graph(
            [node],
            "qlinearadd_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("C", TensorProto.FLOAT, list(c_shape))],
        )
        model = helper.make_model(graph, producer_name="qlinearadd_test")
        quantize_and_verify_with_ort(model, input_names, [a_shape, b_shape], target, dev)

    verify_qlinearadd([4, 2], [4, 2], [4, 2])
    verify_qlinearadd([4, 2], [2], [4, 2])
    verify_qlinearadd([5, 1, 7], [2, 7], [5, 2, 7])


@tvm.testing.parametrize_targets
def test_qlinearmul(target, dev):
    """test_qlinearmul"""

    def verify_qlinearmul(a_shape, b_shape, c_shape):

        _ = np.random.random(a_shape).astype("float32")
        _ = np.random.random(b_shape).astype("float32")

        input_nodes = [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ]
        input_names = [
            "a",
            "b",
        ]

        node = helper.make_node("Mul", input_names, ["C"])
        graph = helper.make_graph(
            [node],
            "qlinearmul_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("C", TensorProto.FLOAT, list(c_shape))],
        )
        model = helper.make_model(graph, producer_name="qlinearmul_test")
        quantize_and_verify_with_ort(model, input_names, [a_shape, b_shape], target, dev)

    verify_qlinearmul([7], [7], [7])
    verify_qlinearmul([4, 2], [4, 2], [4, 2])
    verify_qlinearmul([4, 2], [2], [4, 2])
    verify_qlinearmul([5, 1, 7], [2, 7], [5, 2, 7])


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/11375")
@tvm.testing.parametrize_targets
def test_qlinearleakyrelu(target, dev):
    """test_qlinearleakyrelu"""

    def verify_qlinearleakyrelu(inshape, kwargs):

        in_array = np.random.random(inshape).astype("float32")
        node = helper.make_node("LeakyRelu", ["X"], ["Y"], **kwargs)

        graph = helper.make_graph(
            [node],
            "qlinearRelu_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list(in_array.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(in_array.shape))],
        )
        model = helper.make_model(graph, producer_name="qlinearRelu_test")
        args = (model, ["X"], [in_array.shape], target, dev)
        if dev == "cuda":
            quantize_and_verify_with_ort(*args, rtol=1e-2, atol=1e-2)
        else:
            quantize_and_verify_with_ort(*args)

    verify_qlinearleakyrelu([2, 4, 5, 6], {"alpha": 0.25})
    verify_qlinearleakyrelu([6, 5, 6, 7], {"alpha": 0.35})
    verify_qlinearleakyrelu([5, 1, 4, 6], {"alpha": 0.65})


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/11375")
@tvm.testing.parametrize_targets
def test_qlinearsigmoid(target, dev):
    """test_qlinearsigmoid"""

    def verify_qlinearsigmoid(a_shape):

        _ = np.random.random(a_shape).astype("float32")

        input_nodes = [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))]

        node = helper.make_node("Sigmoid", ["a"], ["B"])
        graph = helper.make_graph(
            [node],
            "qlinearsigmoid_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("B", TensorProto.FLOAT, list(a_shape))],
        )
        model = helper.make_model(graph, producer_name="qlinearsigmoid_test")
        quantize_and_verify_with_ort(model, ["a"], [a_shape], target, dev)

    verify_qlinearsigmoid([4, 2])
    verify_qlinearsigmoid([5])
    verify_qlinearsigmoid([3, 4, 5])
    verify_qlinearsigmoid([])


@tvm.testing.parametrize_targets
def test_qlinearsoftmax(target, dev):
    """test_qlinearsoftmax"""

    def verify_qlinearsoftmax(a_shape):

        _ = np.random.random(a_shape).astype("float32")

        input_nodes = [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))]

        node = helper.make_node("Softmax", ["a"], ["B"])
        graph = helper.make_graph(
            [node],
            "qlinearsoftmax_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("B", TensorProto.FLOAT, list(a_shape))],
        )
        model = helper.make_model(graph, producer_name="qlinearsoftmax_test")
        quantize_and_verify_with_ort(model, ["a"], [a_shape], target, dev)

    verify_qlinearsoftmax([4, 2])
    verify_qlinearsoftmax([5])
    verify_qlinearsoftmax([3, 4, 5])


@tvm.testing.parametrize_targets("llvm")
def test_random_bernoulli(target, dev):
    """test_random_bernoulli"""

    def _get_tinygrad_output(
        inputs,
        out_dtype="int32",
        seed=None,
        target=target,
        dev=dev,
        use_vm=False,
        freeze_params=False,
    ):
        def get_bernoulli_model(shape, in_dtype="float32", out_dtype="int32", seed=None):
            onnx_itype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
            onnx_otype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_dtype)]
            node = helper.make_node(
                "Bernoulli",
                ["input"],
                ["output"],
            )
            dtype_attr = helper.make_attribute("dtype", onnx_otype)
            node.attribute.append(dtype_attr)
            if seed is not None:
                seed_attr = helper.make_attribute("seed", float(seed))
                node.attribute.append(seed_attr)

            graph = helper.make_graph(
                [node],
                "random_bernoulli_test",
                inputs=[helper.make_tensor_value_info("input", onnx_itype, list(shape))],
                outputs=[helper.make_tensor_value_info("output", onnx_otype, list(shape))],
            )
            return helper.make_model(graph, producer_name="random_bernoulli_test")

        shape = inputs.shape
        in_dtype = inputs.dtype
        model = get_bernoulli_model(shape, in_dtype, out_dtype, seed)

        if use_vm:
            return get_tinygrad_output_with_vm(
                model,
                inputs,
                target,
                dev,
                freeze_params=freeze_params,
            )
        else:
            return get_tinygrad_output(
                model,
                inputs,
                target,
                dev,
            )

    def binom_test(input, ideal_mean, threshold=0.05):
        # This test is strictly appropriate when input probabilities are all identical.
        # In that case, it should lead to flaky failures in only one run in a million (p>=1e-6).
        # The test should be over-conservative when input probabilities are not identical.
        # (i.e., It should have a rate of flaky failures lower than one run in a million.)
        # If this test starts repeatedly throwing flaky failures, consult a statistician
        # in addition to your regular debugging.
        bnm_test_res = scipy.stats.binomtest(
            k=np.sum(input, dtype="int32"), n=len(input), p=ideal_mean
        )
        return bnm_test_res.pvalue > threshold

    def verify_bernoulli(
        inputs=None,
        shape=[],
        in_dtype="float32",
        out_dtype="int32",
        seed=None,
        target=target,
        dev=dev,
        use_vm=False,
        freeze_params=False,
        in_out_equal=False,
    ):
        if inputs is None:
            assert len(shape) != 0
            inputs = np.random.uniform(size=shape).astype(in_dtype)

        tvm_out = _get_tinygrad_output(
            inputs,
            out_dtype,
            seed,
            target,
            dev,
            use_vm,
            freeze_params,
        )

        if isinstance(tvm_out, list):
            tvm_out = tvm_out[0]
        # check that values are 0 or 1
        tvm_flat = tvm_out.flatten()
        assert np.array_equal(tvm_flat, tvm_flat.astype("bool"))
        if in_out_equal:
            tvm.testing.assert_allclose(inputs, tvm_out)
        else:
            # check that mean value is close to the theoretical one by binomial test
            ideal_mean = np.mean(inputs)
            repeats = 3
            check = False
            for i in range(repeats):
                if binom_test(tvm_flat, ideal_mean):
                    check = True
                    break
                else:
                    # repeat with new seed
                    seed = np.random.randint(1e6)
                    tvm_flat = _get_tinygrad_output(
                        inputs,
                        out_dtype,
                        seed,
                        target,
                        dev,
                        use_vm,
                        freeze_params,
                    ).flatten()
            assert check, "Binomial test failed"

    # Test input sequence of 0 and 1
    inputs = np.random.randint(2, size=[10000]).astype("float32")
    verify_bernoulli(inputs, in_out_equal=True)

    # Binomial test input with 0.5 values
    val_num = 10000
    inputs = np.ones([val_num], dtype="float32") * 0.5
    verify_bernoulli(inputs)

    # Binomial test input with 0.1 values
    inputs = np.ones([val_num], dtype="float32") * 0.1
    verify_bernoulli(inputs)

    # Simple test
    verify_bernoulli(shape=[val_num])

    # Floating output type
    verify_bernoulli(shape=[val_num], out_dtype="float32")

    # Double input type
    verify_bernoulli(shape=[val_num], in_dtype="float64")

    # Test N-D tensor generation
    verify_bernoulli(shape=[2, 4, 100, 100])

    # Test with seed
    verify_bernoulli(shape=[val_num], seed=np.random.randint(1e6))

    # Test result determinism with the same seeds
    inputs = np.random.uniform(size=[val_num])
    fixed_seed = np.random.randint(1e6)
    tvm_out_1 = _get_tinygrad_output(inputs, seed=fixed_seed)
    tvm_out_2 = _get_tinygrad_output(inputs, seed=fixed_seed)
    tvm.testing.assert_allclose(tvm_out_1, tvm_out_2)


@tvm.testing.parametrize_targets("llvm")
def test_random_uniform(target, dev):
    """test_random_uniform"""

    def get_random_uniform(shape, dtype="float32", high=1.0, low=0.0, seed=None):
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        node = helper.make_node(
            "RandomUniform", [], ["out"], shape=shape, dtype=ONNX_DTYPE, high=high, low=low
        )
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "random_uniform_test",
            inputs=[],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_uniform_test")
        return get_tinygrad_output_with_vm(
            model,
            [],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Check that function runs and produces proper shape.
    vals = get_random_uniform([10], dtype="float32")
    assert list(vals.shape) == [10]
    assert vals.dtype == "float32"

    # Test N-D tensor generation.
    vals = get_random_uniform([1, 3, 100, 100], dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]

    # Check that bounds aren't exceeded.
    vals = get_random_uniform(shape=[100], high=100.0, low=-100.0)
    assert list(vals.shape) == [100]
    assert all(vals >= -100) and all(vals <= 100)

    # Check that a fixed seed produces the same values when run twice.
    vals_1 = get_random_uniform(shape=[10], seed=1)
    vals_2 = get_random_uniform(shape=[10], seed=1)
    assert all(vals_1 == vals_2)

    # Test against an expected output with a fixed seed.
    real = get_random_uniform(shape=[10], seed=5.0)
    expected = np.asarray(
        [
            0.043976,
            0.96656,
            0.292199,
            0.904297,
            0.25167,
            0.521778,
            0.778985,
            0.085463,
            0.939846,
            0.194201,
        ]
    )
    tvm.testing.assert_allclose(real, expected, rtol=1e-5)


@tvm.testing.parametrize_targets("llvm")
def test_random_uniform_like(target, dev):
    """test_random_uniform_like"""

    def get_random_uniform_like(input_, shape, dtype=None, high=1.0, low=0.0, seed=None):
        node = helper.make_node("RandomUniformLike", ["in"], ["out"], high=high, low=low)
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        ONNX_DTYPE = None
        if dtype is not None:
            ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            dtype_attr = helper.make_attribute("dtype", ONNX_DTYPE)
            node.attribute.append(dtype_attr)
        else:
            dtype = input_.dtype
            ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]

        graph = helper.make_graph(
            [node],
            "random_uniform_test",
            inputs=[helper.make_tensor_value_info("in", ONNX_DTYPE, shape)],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_uniform_like_test")
        return get_tinygrad_output_with_vm(
            model,
            [input_],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Check that function runs and produces proper shape and dtype.
    shape = [10]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_uniform_like(input_array, shape, dtype="float32")
    assert list(vals.shape) == [10]
    assert vals.dtype == "float32"

    # Test N-D tensor generation.
    shape = [1, 3, 100, 100]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_uniform_like(input_array, shape, dtype="float64")
    assert list(vals.shape) == shape
    assert vals.dtype == "float64"

    # Check that bounds aren't exceeded.
    shape = [100]
    input_array = np.random.random(shape).astype("float64")
    vals = get_random_uniform_like(input_array, shape, high=100.0, low=-100.0)
    assert list(vals.shape) == shape
    assert all(vals >= -100) and all(vals <= 100)

    # Test against an expected output with a fixed seed.
    shape = [10]
    input_array = np.random.random(shape).astype("float32")
    real = get_random_uniform_like(input_array, shape=[10], seed=5.0)
    expected = np.asarray(
        [
            0.043976,
            0.96656,
            0.292199,
            0.904297,
            0.25167,
            0.521778,
            0.778985,
            0.085463,
            0.939846,
            0.194201,
        ]
    )
    tvm.testing.assert_allclose(real, expected, rtol=1e-5)


@tvm.testing.parametrize_targets("llvm")
def test_random_normal(target, dev):
    """test_random_normal"""

    def get_random_normal(shape, dtype="float32", scale=1.0, mean=0.0, seed=None):
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        node = helper.make_node(
            "RandomNormal", [], ["out"], shape=shape, dtype=ONNX_DTYPE, scale=scale, mean=mean
        )
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "random_normal_test",
            inputs=[],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_normal_test")
        return get_tinygrad_output_with_vm(
            model,
            [],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Test N-D tensor generation.
    vals = get_random_normal([1, 3, 100, 100], dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 0.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 1.0, rtol=0.1, atol=0.1)

    # Test mean=2.0 scale=10.0
    vals = get_random_normal([1, 3, 100, 100], mean=2.0, scale=10.0, dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 2.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 10.0, rtol=0.1, atol=0.1)

    # Check that a fixed seed produces the same values when run twice.
    vals_1 = get_random_normal(shape=[10], seed=1.0)
    vals_2 = get_random_normal(shape=[10], seed=1.0)
    assert all(vals_1 == vals_2)


@tvm.testing.parametrize_targets("llvm")
def test_random_normal_like(target, dev):
    """test_random_normal_like"""

    def get_random_normal_like(input_, shape, dtype="float32", scale=1.0, mean=0.0, seed=None):
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        node = helper.make_node(
            "RandomNormalLike", ["in"], ["out"], dtype=ONNX_DTYPE, scale=scale, mean=mean
        )
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "random_normal_like_test",
            inputs=[helper.make_tensor_value_info("in", ONNX_DTYPE, shape)],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_normal_like_test")
        return get_tinygrad_output_with_vm(
            model,
            [input_],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Test N-D tensor generation.
    shape = [1, 3, 100, 100]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_normal_like(input_array, [1, 3, 100, 100], dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 0.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 1.0, rtol=0.1, atol=0.1)

    # Test mean=2.0 scale=10.0
    shape = [1, 3, 100, 100]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_normal_like(
        input_array, [1, 3, 100, 100], mean=2.0, scale=10.0, dtype="float32"
    )
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 2.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 10.0, rtol=0.1, atol=0.1)


@tvm.testing.parametrize_targets("llvm")
def test_multinomial(target, dev):
    def get_multinomial(input, shape, sample_size, seed=None):
        IN_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("float32")]
        OUT_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("int32")]
        node = helper.make_node("Multinomial", ["in"], ["out"], sample_size=sample_size)
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "multinomial_test",
            inputs=[helper.make_tensor_value_info("in", IN_DTYPE, shape)],
            outputs=[helper.make_tensor_value_info("out", OUT_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="multinomial_test")
        return get_tinygrad_output_with_vm(
            model,
            [input],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Test N-D tensor generation.
    shape = [3]
    sample_size = 2
    probs = np.random.random(shape).astype("float32")
    indices = get_multinomial(probs, shape, sample_size)
    # Since specific values are random, we'll check that the output shape is
    # correct and the values chosen are all valid indices.
    assert list(indices.shape) == [sample_size]
    assert np.max(indices) < shape[-1]

    # Test 2d multinomial
    shape = [10, 5]
    sample_size = 4
    probs = np.random.random(shape).astype("float32")
    indices = get_multinomial(probs, shape, sample_size)
    assert list(indices.shape) == [10, sample_size]
    assert np.max(indices) < shape[-1]


@tvm.testing.parametrize_targets
def test_convinteger(target, dev):
    """test_convinteger"""

    def verify_convinteger(
        x_shape,
        w_shape,
        y_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        auto_pad="NOTSET",
        dtype="uint8",
    ):
        x_array = np.random.randint(low=0, high=255, size=x_shape).astype(dtype)
        w_array = np.random.uniform(low=0, high=255, size=w_shape).astype(dtype)
        x_zero_point_array = np.random.randint(0, 255, size=[1]).astype(dtype)
        w_zero_point_array = np.random.randint(0, 255, size=[1]).astype(dtype)

        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        input_nodes = [
            helper.make_tensor_value_info("x", ONNX_DTYPE, list(x_shape)),
            helper.make_tensor_value_info("w", ONNX_DTYPE, list(w_shape)),
        ]
        initializer = [
            helper.make_tensor("x_zero_point", ONNX_DTYPE, [], x_zero_point_array),
            helper.make_tensor("w_zero_point", ONNX_DTYPE, [], w_zero_point_array),
        ]
        input_names = ["x", "w", "x_zero_point", "w_zero_point"]
        input_values = [x_array, w_array]

        if padding is None:
            ## autopadding with unset default attributes
            kwargs = {}
            if not all(list(s == 1 for s in strides)):
                kwargs["strides"] = strides
            if not all(list(d == 1 for d in dilations)):
                kwargs["dilations"] = dilations

            node = helper.make_node(
                "ConvInteger",
                inputs=input_names,
                outputs=["y"],
                # Default values for other attributes:
                auto_pad=auto_pad,
                **kwargs,
            )
        else:
            node = helper.make_node(
                "ConvInteger",
                inputs=input_names,
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                # groups=1
                pads=padding,
            )

        graph = helper.make_graph(
            [node],
            "convinteger_test",
            inputs=input_nodes,
            initializer=initializer,
            outputs=[helper.make_tensor_value_info("y", TensorProto.INT32, list(y_shape))],
        )
        model = helper.make_model(graph, producer_name="convinteger_test")
        # opt_level=1 will cause error
        verify_with_ort_with_inputs(model, input_values, target=target, dev=dev, opt_level=2)

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    # only support 2D ConvInteger because we only support qnn.conv2d for now.
    dims = 2

    # Convolution with padding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )

    # Convolution with asymmetric padding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(4, dims),
        repeat(0, dims) + repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution without padding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        2 * repeat(0, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution with autopadding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with valid autopadding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="VALID",
    )
    # Convolution with non uniform stride
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(2, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with dilation
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(2, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(2, dims),
    )


@tvm.testing.parametrize_targets
def test_bitshift(target, dev):
    """test_bitshift"""

    def verify_bitshift(in_shape, shift_shape, high=1000000000, in_dtype="uint64"):
        in_shape = list(in_shape)
        shift_shape = list(shift_shape)

        # Create an input for each tensor.
        tensor_values = [
            np.random.randint(high, size=in_shape).astype(in_dtype),
            np.random.randint(16, size=shift_shape).astype(in_dtype),
            np.random.randint(16, size=shift_shape).astype(in_dtype),
        ]

        bitshift_left_node = helper.make_node(
            "BitShift",
            inputs=["input", "shift_left"],
            outputs=["shifted"],
            direction="LEFT",
        )

        bitshift_right_node = helper.make_node(
            "BitShift",
            inputs=["shifted", "shift_right"],
            outputs=["output"],
            direction="RIGHT",
        )

        # Create input and output tensors.
        proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
        graph_inputs = [
            helper.make_tensor_value_info("input", proto_type, in_shape),
            helper.make_tensor_value_info("shift_left", proto_type, shift_shape),
            helper.make_tensor_value_info("shift_right", proto_type, shift_shape),
        ]

        graph_outputs = [helper.make_tensor_value_info("output", proto_type, in_shape)]

        graph_nodes = [bitshift_left_node, bitshift_right_node]

        graph = helper.make_graph(
            graph_nodes,
            "BitShift_test",
            inputs=graph_inputs,
            outputs=graph_outputs,
        )
        model = helper.make_model(
            graph,
            producer_name="BitShift_test",
        )

        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    shape = (100, 4, 2)
    broadcast_shape = (100, 1, 1)
    # Common bitwise test
    verify_bitshift(shape, shape)
    # Bitwise test with broadcasting
    verify_bitshift(shape, broadcast_shape)


# TODO(vvchernov): return test back than ONNX Runtime in CI will support domain version of 18
@pytest.mark.skip("Currently ONNX Runtime in CI does not support domain version of 18")
@tvm.testing.parametrize_targets
def test_bitwise(target, dev):
    """test_bitwise"""

    def verify_bitwise_ops(A_shape, B_shape, C_shape, D_shape, high=128, in_dtype="int32"):
        A_shape = list(A_shape)
        B_shape = list(B_shape)
        C_shape = list(C_shape)
        D_shape = list(D_shape)

        # Create an input for each tensor.
        tensor_values = [
            np.random.randint(high, size=A_shape).astype(in_dtype),
            np.random.randint(high, size=B_shape).astype(in_dtype),
            np.random.randint(high, size=C_shape).astype(in_dtype),
            np.random.randint(high, size=D_shape).astype(in_dtype),
        ]

        or_node = helper.make_node(
            "BitwiseOr",
            inputs=["A", "B"],
            outputs=["OR"],
        )

        and_node = helper.make_node(
            "BitwiseAnd",
            inputs=["OR", "C"],
            outputs=["AND"],
        )

        xor_node = helper.make_node(
            "BitwiseXor",
            inputs=["AND", "D"],
            outputs=["XOR"],
        )

        not_node = helper.make_node(
            "BitwiseNot",
            inputs=["XOR"],
            outputs=["output"],
        )

        # Create input and output tensors.
        proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
        graph_inputs = [
            helper.make_tensor_value_info("A", proto_type, A_shape),
            helper.make_tensor_value_info("B", proto_type, B_shape),
            helper.make_tensor_value_info("C", proto_type, C_shape),
            helper.make_tensor_value_info("D", proto_type, D_shape),
        ]

        graph_outputs = [
            helper.make_tensor_value_info("output", proto_type, A_shape),
        ]

        graph_nodes = [
            or_node,
            and_node,
            xor_node,
            not_node,
        ]

        graph = helper.make_graph(
            graph_nodes,
            "Bitwise_test",
            inputs=graph_inputs,
            outputs=graph_outputs,
        )
        model = helper.make_model(
            graph,
            producer_name="Bitwise_test",
        )

        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    shape = (100, 4, 2)
    broadcast_shape = (100, 1, 1)
    dtypes = ["int8", "uint8", "int32", "uint32"]
    high_vals = [128, 128, 2147483648, 2147483648]
    for high, dtype in zip(high_vals, dtypes):
        # Common bitwise test
        verify_bitwise_ops(shape, shape, shape, shape, high, dtype)
        # Bitwise test with broadcasting
        verify_bitwise_ops(shape, broadcast_shape, broadcast_shape, broadcast_shape, high, dtype)


@tvm.testing.parametrize_targets
def test_scan(target, dev):
    """test_scan"""

    def verify_scan(
        input_shapes,
        output_shapes,
        num_scan_inputs,
        scan_input_axes,
        scan_input_directions,
        scan_output_axes,
        scan_output_directions,
        opset,
    ):

        body_input_shapes = copy.deepcopy(input_shapes)
        num_state_inputs = len(input_shapes) - num_scan_inputs

        if opset == 8:
            for i in range(len(input_shapes)):
                body_input_shapes[i].pop(0)
            for i in range(num_state_inputs, len(input_shapes)):
                body_input_shapes[i].pop(0)
        else:
            for i in range(num_state_inputs, len(input_shapes)):
                body_input_shapes[i].pop(scan_input_axes[i - num_state_inputs])

        initial0 = onnx.helper.make_tensor_value_info(
            "initial0", onnx.TensorProto.FLOAT, body_input_shapes[0]
        )
        initial1 = onnx.helper.make_tensor_value_info(
            "initial1", onnx.TensorProto.FLOAT, body_input_shapes[1]
        )
        input0 = onnx.helper.make_tensor_value_info(
            "input0", onnx.TensorProto.FLOAT, body_input_shapes[2]
        )
        input1 = onnx.helper.make_tensor_value_info(
            "input1", onnx.TensorProto.FLOAT, body_input_shapes[3]
        )
        input2 = onnx.helper.make_tensor_value_info(
            "input2", onnx.TensorProto.FLOAT, body_input_shapes[4]
        )
        state0 = onnx.helper.make_tensor_value_info(
            "state0", onnx.TensorProto.FLOAT, body_input_shapes[0]
        )
        scan_out0 = onnx.helper.make_tensor_value_info(
            "scan_out0", onnx.TensorProto.FLOAT, body_input_shapes[0]
        )
        state1 = onnx.helper.make_tensor_value_info(
            "state1", onnx.TensorProto.FLOAT, body_input_shapes[1]
        )
        scan_out1 = onnx.helper.make_tensor_value_info(
            "scan_out1", onnx.TensorProto.FLOAT, body_input_shapes[1]
        )
        add_node = onnx.helper.make_node(
            "Add",
            inputs=["initial0", "input0"],
            outputs=["state0"],
        )
        id_node_0 = onnx.helper.make_node(
            "Identity",
            inputs=["state0"],
            outputs=["scan_out0"],
        )
        matmul_node = onnx.helper.make_node(
            "MatMul",
            inputs=["input1", "input2"],
            outputs=["matmul_out"],
        )
        sub_node = onnx.helper.make_node(
            "Sub",
            inputs=["initial1", "matmul_out"],
            outputs=["state1"],
        )
        id_node_1 = onnx.helper.make_node(
            "Identity",
            inputs=["state1"],
            outputs=["scan_out1"],
        )
        scan_body = onnx.helper.make_graph(
            [add_node, id_node_0, matmul_node, sub_node, id_node_1],
            "scan_body",
            [initial0, initial1, input0, input1, input2],
            [state0, state1, scan_out0, scan_out1],
        )
        # create scan op node
        scan_node = None
        if opset == 8:
            scan_node = onnx.helper.make_node(
                "Scan",
                inputs=["", "init0", "init1", "in0", "in1", "in2"],
                outputs=["s0", "s1", "scan0", "scan1"],
                num_scan_inputs=num_scan_inputs,
                body=scan_body,
            )
        else:
            scan_node = onnx.helper.make_node(
                "Scan",
                inputs=["init0", "init1", "in0", "in1", "in2"],
                outputs=["s0", "s1", "scan0", "scan1"],
                num_scan_inputs=num_scan_inputs,
                body=scan_body,
                scan_input_axes=scan_input_axes,
                scan_input_directions=scan_input_directions,
                scan_output_axes=scan_output_axes,
                scan_output_directions=scan_output_directions,
            )
        input_info = [
            helper.make_tensor_value_info("init0", TensorProto.FLOAT, input_shapes[0]),
            helper.make_tensor_value_info("init1", TensorProto.FLOAT, input_shapes[1]),
            helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shapes[2]),
            helper.make_tensor_value_info("in1", TensorProto.FLOAT, input_shapes[3]),
            helper.make_tensor_value_info("in2", TensorProto.FLOAT, input_shapes[4]),
        ]
        out_info = [
            helper.make_tensor_value_info("s0", TensorProto.FLOAT, output_shapes[0]),
            helper.make_tensor_value_info("s1", TensorProto.FLOAT, output_shapes[1]),
            helper.make_tensor_value_info("scan0", TensorProto.FLOAT, output_shapes[2]),
            helper.make_tensor_value_info("scan1", TensorProto.FLOAT, output_shapes[3]),
        ]
        graph = helper.make_graph(
            nodes=[scan_node],
            name="scan_test",
            inputs=input_info,
            outputs=out_info,
        )
        model = onnx.helper.make_model(graph, producer_name="scan-test")
        init0 = np.random.uniform(low=0, high=255, size=input_shapes[0]).astype(np.float32)
        init1 = np.random.uniform(low=0, high=255, size=input_shapes[1]).astype(np.float32)
        in0 = np.random.uniform(low=0, high=255, size=input_shapes[2]).astype(np.float32)
        in1 = np.random.uniform(low=0, high=255, size=input_shapes[3]).astype(np.float32)
        in2 = np.random.uniform(low=0, high=255, size=input_shapes[4]).astype(np.float32)
        input_values = [init0, init1, in0, in1, in2]

        verify_with_ort_with_inputs(
            model,
            input_values,
            target=target,
            dev=dev,
            opt_level=2,
            use_vm=True,
            opset=opset,
        )

    # opset 8
    input_shapes = [[2, 6, 7, 8], [2, 3, 3], [2, 5, 6, 7, 8], [2, 5, 3, 4], [2, 5, 4, 3]]
    output_shapes = [[2, 6, 7, 8], [2, 3, 3], [2, 5, 6, 7, 8], [2, 5, 3, 3]]
    # input_shapes, output_shapes, num_scan_inputs, scan_input_axes, scan_input_directions,
    # scan_output_axes, scan_output_directions, opset
    verify_scan(input_shapes, output_shapes, 3, [0] * 3, [0] * 3, [0] * 2, [0] * 2, 8)
    # opset 9
    input_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [5, 3, 4], [5, 4, 3]]
    output_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [5, 3, 3]]
    verify_scan(input_shapes, output_shapes, 3, [0] * 3, [0] * 3, [0] * 2, [0] * 2, 9)

    input_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [3, 4, 5], [4, 5, 3]]
    output_shapes = [[6, 7, 8], [3, 3], [6, 5, 7, 8], [3, 5, 3]]
    verify_scan(input_shapes, output_shapes, 3, [0, 2, 1], [1] * 3, [1] * 2, [1] * 2, 9)
    # Negative axes
    input_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [3, 4, 5], [4, 5, 3]]
    output_shapes = [[6, 7, 8], [3, 3], [6, 5, 7, 8], [3, 5, 3]]
    verify_scan(input_shapes, output_shapes, 3, [-4, -1, -2], [1] * 3, [-3, -2], [1] * 2, 9)


@tvm.testing.parametrize_targets
def test_linear_regressor(target, dev):
    """test_linear_regressor"""

    def verify_linear_regressor(a_shape, c_shape, i_shape, targets=1, batch=1):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        out_shape = (batch, targets)

        coefficients = np.random.uniform(size=c_shape).astype("float32")
        intercepts = np.random.uniform(size=i_shape).astype("float32")

        mul_node = helper.make_node(
            "LinearRegressor",
            ["a"],
            ["out"],
            coefficients=coefficients,
            intercepts=intercepts,
            targets=targets,
            domain="ai.onnx.ml",
        )

        graph = helper.make_graph(
            [mul_node],
            "LinearRegressor_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="LinearRegressor_test",
            opset_imports=[
                onnx.helper.make_opsetid("ai.onnx.ml", 1),
            ],
        )
        verify_with_ort_with_inputs(model, [a_array], target=target, dev=dev)

    verify_linear_regressor((1, 3), (3), (1))
    verify_linear_regressor((2, 10), (10), (1), batch=2)
    verify_linear_regressor((1, 3), (30), (10), targets=10)
    verify_linear_regressor((10, 3), (30), (10), targets=10, batch=10)
    verify_linear_regressor((1, 4), (3), (1))


@tvm.testing.parametrize_targets
def test_dft(target, dev):
    """test_dft"""

    def verify_dft(
        _axis,
        _inverse,
        _onesided,
        _dft_length,
        _input_shape,
        _output_shape,
    ):
        input_names = ["input"]
        if _dft_length is not None:
            input_names.append("dft_length")

        node = onnx.helper.make_node(
            "DFT",
            inputs=input_names,
            outputs=["output"],
            axis=_axis,
            inverse=_inverse,
            onesided=_onesided,
        )

        nodes = []
        if _dft_length is not None:
            nodes.append(
                make_constant_node("dft_length", TensorProto.INT32, [], [_dft_length]),
            )
        nodes.append(node)

        graph = helper.make_graph(
            nodes,
            "dft_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, _input_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, _output_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="dft_test")

        _input = np.random.normal(size=_input_shape).astype("float32")
        verify_with_ort_with_inputs(
            model,
            [_input],
            [_input_shape],
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
            use_vm=False,
        )

    batch_size = 5
    n = 2
    D = 7

    for axis in list(range(1, n)) + [-2]:
        for inverse, onesided in [(0, 0), (0, 1), (1, 0), (None, None)]:
            for n_fft in [D, D - 1, D + 1]:
                for c in [1, 2]:
                    input_shape = [batch_size] + n * [D] + [c]
                    output_shape = [batch_size] + n * [D] + [2]
                    if onesided == 1:
                        output_shape[axis] = output_shape[axis] // 2 + 1
                    verify_dft(axis, inverse, onesided, n_fft, input_shape, output_shape)


@tvm.testing.parametrize_targets
def test_sequence(target, dev):
    """test_sequence"""

    def verify_sequence_ops(tensor_shape, num_tensors, axis=0, position=0, new_axis=None):
        tensor_shape = list(tensor_shape)
        tensor_values = []
        for i in range(num_tensors):
            tensor_values.append(np.random.uniform(size=tensor_shape).astype("float32"))

        # Create an input for each tensor.
        input_tensor_names = []
        for i in range(num_tensors):
            name = f"input_tensor_{i}"
            input_tensor_names.append(name)

        # Test creating a tensor sequence.
        construct_node = helper.make_node(
            "SequenceConstruct",
            inputs=input_tensor_names,
            outputs=["sequence"],
        )

        position_node = make_constant_node("position", TensorProto.INT32, (), [position])

        # Test sequence insertion.
        insert_node = helper.make_node(
            "SequenceInsert",
            inputs=["sequence", input_tensor_names[0], "position"],
            outputs=["inserted_sequence"],
        )

        # Test sequence erase.
        erase_node = helper.make_node(
            "SequenceErase",
            inputs=["inserted_sequence", "position"],
            outputs=["erased_sequence"],
        )

        # Test sequence concatenation.
        concat_node = helper.make_node(
            "ConcatFromSequence",
            inputs=["erased_sequence"],
            outputs=["concat_sequence"],
            axis=axis,
        )

        # Test splitting a tensor into a sequence.
        split_node = helper.make_node(
            "SplitToSequence", inputs=["concat_sequence"], outputs=["split_sequence"], axis=axis
        )

        # Test tensor extraction from sequence
        at_node = helper.make_node(
            "SequenceAt", inputs=["split_sequence", "position"], outputs=["output"]
        )

        # Test sequence length
        length_node = helper.make_node(
            "SequenceLength", inputs=["split_sequence"], outputs=["output_2"]
        )

        if new_axis is not None:
            new_axis_attr = helper.make_attribute("new_axis", new_axis)
            concat_node.attribute.append(new_axis_attr)

        # Create input and output tensors.
        graph_inputs = []
        for name in input_tensor_names:
            input_tensor = helper.make_tensor_value_info(name, TensorProto.FLOAT, tensor_shape)
            graph_inputs.append(input_tensor)

        # Construct output tensor.
        output_shape = tensor_shape
        if new_axis is not None:
            output_shape.insert(axis, 1)
            output_shape[axis] = num_tensors + 1
        else:
            output_shape[axis] = (num_tensors + 1) * output_shape[axis]
        graph_outputs = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
            helper.make_tensor_value_info("output_2", TensorProto.INT64, []),
        ]

        graph_nodes = [
            position_node,
            construct_node,
            insert_node,
            erase_node,
            concat_node,
            split_node,
            at_node,
            length_node,
        ]

        graph = helper.make_graph(
            graph_nodes,
            "Sequence_test",
            inputs=graph_inputs,
            outputs=graph_outputs,
        )
        model = helper.make_model(
            graph,
            producer_name="Sequence_test",
        )

        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    verify_sequence_ops((10, 3), 2)
    verify_sequence_ops((3, 3, 3, 3), 4, position=3)
    verify_sequence_ops((3, 3, 3, 3), 4, axis=2)
    verify_sequence_ops((3, 3, 3, 3), 4, axis=2, new_axis=1)


@tvm.testing.parametrize_targets
def test_empty_sequence(target, dev):
    """test_empty_sequence"""

    # Test creating an empty tensor sequence.
    empty_node = helper.make_node(
        "SequenceEmpty",
        inputs=[],
        outputs=["empty_sequence"],
    )

    length_node = helper.make_node("SequenceLength", inputs=["empty_sequence"], outputs=["output"])

    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.INT64, [])]

    graph_nodes = [empty_node, length_node]

    graph = helper.make_graph(
        graph_nodes,
        "Sequence_empty_test",
        inputs=[],
        outputs=graph_outputs,
    )

    model = helper.make_model(
        graph,
        producer_name="Sequence_empty_test",
    )

    verify_with_ort_with_inputs(model, [], target=target, dev=dev)


def test_exporting_node_renamed_model():
    """test exproting model when export_node_renamed_model is set"""

    a_name, a_shape = "a", (4, 3)
    b_name, b_shape = "b", (3, 4)
    out_name, out_shape = "out", [a_shape[0], b_shape[1]]
    temp_dir = utils.tempdir().path

    # model definition
    mul_node = helper.make_node("MatMul", [a_name, b_name], [out_name])
    graph = helper.make_graph(
        [mul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info(a_name, TensorProto.FLOAT, a_shape),
            helper.make_tensor_value_info(b_name, TensorProto.FLOAT, b_shape),
        ],
        outputs=[helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="matmul_test")

    # get frontend model
    shape_dict = {a_name: a_shape, b_name: b_shape}
    _, _ = relay.frontend.from_onnx(model, shape_dict, export_node_renamed_model_path=temp_dir)

    exported_model_name = os.listdir(temp_dir)[0]
    assert "tvm_exported_model_" in exported_model_name

    exported_model = onnx.load(os.path.join(temp_dir, exported_model_name))
    assert exported_model.graph.node[0].name == "MatMul_0"


class TestSetSpan:
    """test structural equal between translated / hand-crafted relay IR with span tagged."""

    def _verify(self, res_fptr, golden_fptr):
        with tvm.testing.enable_span_filling():
            with_span = res_fptr()
        with tvm.testing.disable_span_filling():
            without_span = res_fptr()
        tvm.ir.assert_structural_equal(with_span, without_span)
        _verify_structural_equal_with_span(with_span, golden_fptr())

    def test_conv2d_bias_add_span(self):
        padding = [0, 0, 0, 0]
        k_shape = [7, 7]
        y_shape, y_name = [1, 6, 10, 10], "y"
        x_shape, x_name = [1, 3, 10, 10], "x"
        b_shape, b_name = [6], "b"
        b_val = np.random.random(b_shape).astype(np.float32)
        w_shape, w_name = [6, 3, 7, 7], "w"
        w_val = np.random.random(w_shape).astype(np.float32)
        group, strides, dilations = 1, [1, 1], [1, 1]
        conv_name = "conv2d"

        def _res():
            # model definition
            node = helper.make_node(
                "Conv",
                inputs=[x_name, w_name, b_name],
                outputs=[y_name],
                kernel_shape=k_shape,
                strides=strides,
                dilations=dilations,
                group=group,
                pads=padding,
                name=conv_name,
            )
            graph = helper.make_graph(
                [node],
                "conv_test",
                inputs=[helper.make_tensor_value_info(x_name, TensorProto.FLOAT, x_shape)],
                outputs=[helper.make_tensor_value_info(y_name, TensorProto.FLOAT, y_shape)],
                initializer=[
                    helper.make_tensor(
                        w_name,
                        TensorProto.FLOAT,
                        dims=w_shape,
                        vals=w_val.flatten(),
                    ),
                    helper.make_tensor(
                        b_name,
                        TensorProto.FLOAT,
                        dims=b_shape,
                        vals=b_val.flatten(),
                    ),
                ],
            )
            model = helper.make_model(graph, producer_name="conv_test")

            # get frontend model
            shape_dict = {x_name: x_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            conv_si = conv_name
            x = relay.var(
                x_name,
                shape=tuple(x_shape),
                span=_create_span(f"{conv_si}.{x_name}"),
            )
            conv_weight = relay.const(
                w_val,
                span=_create_span(f"{conv_si}.{w_name}"),
            )
            conv_bias = relay.const(
                b_val,
                span=_create_span(f"{conv_si}.{b_name}"),
            )
            conv_out = _set_span(
                relay.nn.conv2d(
                    x,
                    conv_weight,
                    padding=[0] * 4,
                    channels=y_shape[1],
                    kernel_size=k_shape,
                ),
                conv_si,
            )
            bias_out = _set_span(relay.nn.bias_add(conv_out, conv_bias), conv_si)
            return infer_type(relay.Function([x], bias_out))

        self._verify(_res, _golden)

    def test_batchnorm_span(self):
        input_name, in_shape = "x", [1, 16, 10, 10]
        bn_name = "bn"
        output_name = "y"
        scale_name = "scale"
        bias_name = "b"
        mean_name = "mean"
        var_name = "var"

        def _res():
            # model definition
            batchnorm = onnx.helper.make_node(
                "BatchNormalization",
                inputs=[input_name, scale_name, bias_name, mean_name, var_name],
                outputs=[output_name],
                name=bn_name,
            )
            graph = helper.make_graph(
                [batchnorm],
                "batchnorm_test",
                inputs=[
                    helper.make_tensor_value_info(input_name, TensorProto.FLOAT, in_shape),
                    helper.make_tensor_value_info(scale_name, TensorProto.FLOAT, [in_shape[1]]),
                    helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, [in_shape[1]]),
                    helper.make_tensor_value_info(mean_name, TensorProto.FLOAT, [in_shape[1]]),
                    helper.make_tensor_value_info(var_name, TensorProto.FLOAT, [in_shape[1]]),
                ],
                outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, in_shape)],
            )
            model = helper.make_model(graph, producer_name="batchnorm_test")

            # get frontend model
            shape_dict = {input_name: in_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            bn_si = bn_name
            x = relay.var(
                input_name,
                shape=tuple(in_shape),
                span=_create_span(f"{bn_si}.{input_name}"),
            )
            bn_scale = relay.var(
                scale_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{scale_name}"),
            )
            bn_bias = relay.var(
                bias_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{bias_name}"),
            )
            bn_rm = relay.var(
                mean_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{mean_name}"),
            )
            bn_rv = relay.var(
                var_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{var_name}"),
            )
            bn_out = _set_span(
                relay.nn.batch_norm(x, bn_scale, bn_bias, bn_rm, bn_rv),
                bn_si,
            )
            bn_tuple_get_item = _set_span(relay.TupleGetItem(bn_out.tuple_value, 0), bn_si)
            return infer_type(
                relay.Function([x, bn_scale, bn_bias, bn_rm, bn_rv], bn_tuple_get_item)
            )

        self._verify(_res, _golden)

    def test_reshape_span(self):
        input_shape = [2, 1, 10, 1, 10]
        new_shape = [2, 1, 10, 10]
        input_name = "in"
        output_name = "out"
        ref_name = "ref_in"
        const_name = "const"
        reshape_name = "reshape"

        def _res():
            # model definition
            ref_array = np.array(new_shape)
            ref_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[ref_name],
                value=helper.make_tensor(
                    name="const_tensor",
                    data_type=TensorProto.INT32,
                    dims=ref_array.shape,
                    vals=ref_array.flatten().astype(int),
                ),
                name=const_name,
            )
            reshape_node = helper.make_node(
                "Reshape",
                [input_name, ref_name],
                [output_name],
                name=reshape_name,
            )
            graph = helper.make_graph(
                [ref_node, reshape_node],
                "reshape_test",
                inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)],
                outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, new_shape)],
            )
            model = helper.make_model(graph, producer_name="reshape_test")

            # get frontend model
            shape_dict = {input_name: input_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            reshape_si = reshape_name
            x = relay.var(
                input_name,
                shape=tuple(input_shape),
                span=_create_span(f"{reshape_si}.{input_name}"),
            )
            reshape_out = _set_span(
                relay.reshape(x, newshape=new_shape),
                reshape_si,
            )
            return infer_type(relay.Function([x], reshape_out))

        self._verify(_res, _golden)

    def test_matmul_span(self):
        a_name, a_shape = "a", (4, 3)
        b_name, b_shape = "b", (3, 4)
        out_name, out_shape = "out", [a_shape[0], b_shape[1]]
        matmul_name = "matmul"

        def _res():
            # model definition
            mul_node = helper.make_node("MatMul", [a_name, b_name], [out_name], name=matmul_name)
            graph = helper.make_graph(
                [mul_node],
                "matmul_test",
                inputs=[
                    helper.make_tensor_value_info(a_name, TensorProto.FLOAT, a_shape),
                    helper.make_tensor_value_info(b_name, TensorProto.FLOAT, b_shape),
                ],
                outputs=[helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
            )
            model = helper.make_model(graph, producer_name="matmul_test")

            # get frontend model
            shape_dict = {a_name: a_shape, b_name: b_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            matmul_si = matmul_name
            a = relay.var(
                a_name,
                shape=tuple(a_shape),
                span=_create_span(f"{matmul_si}.{a_name}"),
            )
            b = relay.var(
                b_name,
                shape=tuple(b_shape),
                span=_create_span(f"{matmul_si}.{b_name}"),
            )
            b_t = _set_span(relay.transpose(b, axes=[1, 0]), matmul_si)
            matmul_out = _set_span(
                relay.nn.dense(a, b_t, out_dtype="float32"),
                matmul_si,
            )
            return infer_type(relay.Function([a, b], matmul_out))

        self._verify(_res, _golden)


@tvm.testing.parametrize_targets
def test_pad_constant_value(target, dev):
    """test_pad_constant_value"""

    def verify_pad_constant_value(constant_value):
        tensor_shape = [1, 2, 257, 126]
        tensor_values = [np.random.uniform(size=tensor_shape).astype("float32")]
        graph_inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, tensor_shape)]
        graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]
        pads = helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 0, 2, 0, 0, 0, 0])
        pad_node = helper.make_node(
            "Pad", ["input", "pads", constant_value], ["output"], mode="constant"
        )
        graph_nodes = [pad_node]
        graph = helper.make_graph(
            graph_nodes,
            "test_pad_constant_value",
            inputs=graph_inputs,
            outputs=graph_outputs,
            initializer=[pads],
        )
        model = helper.make_model(
            graph,
            producer_name="test_pad_constant_value",
        )
        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    verify_pad_constant_value("")


if __name__ == "__main__":
    tvm.testing.main()