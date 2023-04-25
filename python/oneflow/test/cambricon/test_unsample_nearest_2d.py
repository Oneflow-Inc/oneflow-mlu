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


def _test_unsample_nearest_2d(test_case, shape, dtype, device):
    np_arr = np.random.randn(*shape)
    mlu_out = flow._C.upsample_nearest_2d(
        flow.tensor(np_arr, dtype=dtype, device=flow.device(device)),
        height_scale=2.0,
        width_scale=1.5,
    )
    cpu_out = flow._C.upsample_nearest_2d(
        flow.tensor(np_arr, dtype=dtype, device="cpu"),
        height_scale=2.0,
        width_scale=1.5,
    )
    diff = 0.001 if dtype == flow.float16 else 0.0001
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out, diff, diff))


@flow.unittest.skip_unless_1n1d()
class TestUpsampleNearest2DCambriconModule(flow.unittest.TestCase):
    def test_upsample_nearest_2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_unsample_nearest_2d,
        ]
        arg_dict["shape"] = [
            (2, 3, 4, 5,),
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
