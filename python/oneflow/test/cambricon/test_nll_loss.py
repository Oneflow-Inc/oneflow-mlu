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


def _test_nll_loss_forward(test_case, shape, device, dtype):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    C = x.shape[-1]
    N = x.nelement() // x.shape[-1]
    target = flow.randint(0, C, (N,), dtype=flow.int).to(flow.device(device))

    nll = flow.nn.NLLLoss()
    mlu_out = nll(x, target)
    cpu_out = nll(x.cpu(), target.cpu())


@flow.unittest.skip_unless_1n1d()
class TestNLLLossCambriconModule(flow.unittest.TestCase):
    def test_nll_loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_nll_loss_forward,
        ]
        # TODO(WangYi): add more shape after to_contiguous supported
        arg_dict["shape"] = [
            (2, 3,),
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
