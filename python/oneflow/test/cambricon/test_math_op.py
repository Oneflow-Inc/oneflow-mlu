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


def _test_math_op(test_case, op, shape, dtype, device):
    np_arr = np.random.randn(*shape)
    if op == flow.sqrt:
        np_arr = np.clip(np_arr, 0, 10000)
    mlu_out = op(flow.tensor(np_arr, dtype=dtype, device=flow.device(device)))
    cpu_out = op(flow.tensor(np_arr, dtype=dtype, device="cpu"))
    diff = 0.001 if dtype == flow.float16 else 0.0001
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out, diff, diff))


@flow.unittest.skip_unless_1n1d()
class TestMathOpCambriconModule(flow.unittest.TestCase):
    def test_math_op(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_math_op,
        ]
        arg_dict["op"] = [
            flow.cos,
            flow.exp,
            flow.sin,
            flow.sqrt,
        ]
        arg_dict["shape"] = [
            (4,),
            (3, 4,),
            (2, 3, 4,),
        ]
        arg_dict["dtype"] = [
            flow.float32,
            # flow.float16,
        ]
        arg_dict["device"] = ["mlu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
