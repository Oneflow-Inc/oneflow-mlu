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


def _test_l1_l2_regularize_gradient(test_case, shape, device, dtype, optimizer):
    model_np = np.random.randn(*shape)
    model_diff_np = np.random.randn(*shape)
    lr = np.random.uniform()
    weight_decay = np.random.uniform()

    def _get_updated_param(device):
        model = flow.tensor(model_np, device=flow.device(device), dtype=dtype)
        model_diff = flow.tensor(model_diff_np, device=flow.device(device), dtype=dtype)
        model_param = flow.nn.Parameter(model)
        optim = optimizer([model], lr=lr, weight_decay=weight_decay)

        optim.zero_grad()
        (model_param * model_diff).sum().backward()
        optim.step()
        return model_param

    model_updated_cpu = _get_updated_param("cpu")
    model_updated_mlu = _get_updated_param("mlu")

    test_case.assertTrue(np.allclose(model_updated_cpu.numpy(), model_updated_mlu.numpy(), 0.0001, 0.0001))

def _test_functor(test_case, shape, device, dtype):
    model_np = np.random.randn(*shape)
    model_diff_np = np.random.randn(*shape)
    l1 = np.random.uniform()
    l2 = np.random.uniform()


    def _get_result(device):
        if device == "cpu":
            model = flow.tensor(model_np, device=flow.device(device), dtype=flow.float32)
            model_diff = flow.tensor(model_diff_np, device=flow.device(device), dtype=flow.float32)
            result = flow._C.l1_l2_regularize_gradient_test(model, model_diff, l1, l2).float()
        else:
            model = flow.tensor(model_np, device=flow.device(device), dtype=dtype)
            model_diff = flow.tensor(model_diff_np, device=flow.device(device), dtype=dtype)
            result = flow._C.l1_l2_regularize_gradient_test(model, model_diff, l1, l2)
        return result

    result_cpu = _get_result("cpu")
    result_mlu = _get_result("mlu")
    test_case.assertTrue(np.allclose(result_cpu.float().numpy(), result_mlu.cpu().float().numpy(), 0.0001, 0.0001))

@flow.unittest.skip_unless_1n1d()
class TestL1L2RegularizeGradientMLUModule(flow.unittest.TestCase):
    # def test_l1_l2_regularize_gradient(test_case):
    #     arg_dict = OrderedDict()
    #     arg_dict["test_fun"] = [
    #         _test_l1_l2_regularize_gradient,
    #     ]
    #     arg_dict["shape"] = [(200, 200), (400, 400)]
    #     arg_dict["device"] = ["mlu"]
    #     arg_dict["dtype"] = [flow.float16, flow.float32]
    #     arg_dict["optimizer"] = [flow.optim.SGD, flow.optim.Adam, flow.optim.Adagrad]
    #     for arg in GenArgList(arg_dict):
    #         arg[0](test_case, *arg[1:])
    
    def test_functor(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_functor,
        ]
        arg_dict["shape"] = [(200, 200), (400, 400)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float16,
            flow.float32,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])



if __name__ == "__main__":
    unittest.main()
