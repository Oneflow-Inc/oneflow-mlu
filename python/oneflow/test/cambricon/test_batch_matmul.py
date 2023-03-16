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
    return np.random.randn(*shape)


def _test_batch_matmul_forward(test_case, shape, device, dtype):
    """test batch_matmul forward on MLU"""
    a = _get_data(shape[0], dtype)
    b = _get_data(shape[1], dtype)
    # mlu
    mlu_a = flow.tensor(a, device=flow.device(device), dtype=dtype)
    mlu_b = flow.tensor(b, device=flow.device(device), dtype=dtype)
    mlu_out = flow.matmul(mlu_a, mlu_b)
    # cpu
    cpu_a = flow.tensor(a, device=flow.device("cpu"), dtype=dtype)
    cpu_b = flow.tensor(b, device=flow.device("cpu"), dtype=dtype)
    cpu_out = flow.matmul(cpu_a, cpu_b)
    # compare
    diff = 0.0001
    print(cpu_out)
    print(mlu_out)
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), diff, diff))


@flow.unittest.skip_unless_1n1d()
class TestBatchMatmulCambriconModule(flow.unittest.TestCase):
    def test_matmul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_batch_matmul_forward,
        ]
        arg_dict["shape"] = [
            ((2, 3, 4), (2, 4, 5)),
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
