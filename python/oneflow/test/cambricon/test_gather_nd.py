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


def _test_gather_nd(test_case, shape_index, dtype, index_type, device):
    shape, index = shape_index
    np_arr = np.random.randn(*shape)
    cpu_index = flow.tensor(index, dtype=index_type, device=flow.device("cpu"))
    mlu_index = flow.tensor(index, dtype=index_type, device=flow.device(device))
    mlu_out = flow.gather_nd(
        flow.tensor(np_arr, dtype=dtype, device=flow.device(device)), mlu_index
    )
    cpu_out = flow.gather_nd(flow.tensor(np_arr, dtype=dtype, device="cpu"), cpu_index)
    diff = 0.001 if dtype == flow.float16 else 0.0001
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out, diff, diff))


@flow.unittest.skip_unless_1n1d()
class TestGatherNdCambriconModule(flow.unittest.TestCase):
    def test_gather_nd(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_gather_nd,
        ]
        arg_dict["shape_index"] = [
            ((3, 4, 5, 6), [[0, 1], [1, 2]]),
        ]
        arg_dict["dtype"] = [
            flow.float32,
            flow.int32,
            flow.float16,
        ]
        arg_dict["index_type"] = [
            flow.int64,
            flow.int32,
        ]
        arg_dict["device"] = ["mlu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
