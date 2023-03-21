"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _get_data(shape, dtype):
    array = np.random.randn(*shape)
    if dtype == flow.int:
        array = array.astype(int)
    return array


def _test_activation_backward_grad(test_case, op, shape, device, dtype):
    array = _get_data(shape, dtype)
    # mlu
    x_mlu = flow.tensor(
        array, dtype=dtype, device=flow.device(device), requires_grad=True
    )
    y_mlu = op(x_mlu)
    y_mlu.backward(flow.ones(*y_mlu.shape, dtype=dtype, device=flow.device(device)))
    grad_mlu = x_mlu.grad
    # cpu
    x_cpu = flow.tensor(
        array, dtype=dtype, device=flow.device("cpu"), requires_grad=True
    )
    y_cpu = op(x_cpu)
    y_cpu.backward(flow.ones(*y_cpu.shape, dtype=dtype, device=flow.device("cpu")))
    grad_cpu = x_cpu.grad
    # compare
    diff = 0.001 if dtype == flow.float16 else 0.0001
    test_case.assertTrue(np.allclose(grad_mlu.numpy(), grad_cpu.numpy(), diff, diff))


@flow.unittest.skip_unless_1n1d()
class TestActivationBackwardGradCambriconModule(flow.unittest.TestCase):
    def test_activation_backward_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_activation_backward_grad]
        arg_dict["op"] = [
            flow.relu,
            flow.gelu,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["mlu"]
        arg_dict["data_type"] = [flow.float]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
