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

import oneflow_mlu
import oneflow as flow
import oneflow.unittest


def _test_add_forward(test_case, shape, device, dtype):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    y = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestAddCambriconModule(flow.unittest.TestCase):
    def test_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_add_forward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.uint8,
            flow.int32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_0_size_add(test_case):
        x = flow.tensor(1.0, device=flow.device("mlu"), dtype=flow.float32)
        y = flow.tensor(2.0, device=flow.device("mlu"), dtype=flow.float32)
        z = x + y
        test_case.assertTrue(np.allclose(z.numpy(), [3.0], 0.0001, 0.0001))


if __name__ == "__main__":
    unittest.main()
