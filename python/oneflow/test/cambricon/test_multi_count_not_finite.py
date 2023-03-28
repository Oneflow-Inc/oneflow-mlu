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
import os
import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList


class TestMluMultiReduceSumPowAbs(flow.unittest.TestCase):
    def test_multi_reduce_sum_pow_abs(test_case):
        cpu_x = []
        mlu_x = []
        for i in range(100):
            x = flow.randn(10 * i) / 0
            cpu_x.append(x)
            mlu_x.append(x.to("mlu"))

        cpu_y = flow._C.multi_count_not_finite(cpu_x)
        mlu_y = flow._C.multi_count_not_finite(mlu_x)
        print(cpu_y, mlu_y)


if __name__ == "__main__":
    unittest.main()
