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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: "oneflow.normalization"

import unittest
import numpy as np

import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_NORMALIZATION_OPS"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"

import oneflow_mlu
import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig


def do_normalization_relu_graph(test_case):
    def get_bn(fused=True):
        if fused:
            bn = flow.nn.FusedBatchNorm2d(num_features=2, eps=1e-5, momentum=0.1).to(
                "mlu"
            )
        else:
            bn = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1).to("mlu")
        bn.eval()
        return bn

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = get_bn()
            self.m.eval()

        def build(self, x):
            return self.m(x)

    class GraphToRunWithOpt(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = get_bn(fused=False)
            self.m.eval()

        def build(self, x):
            return flow.relu(self.m(x))

    graph_to_run = GraphToRun()
    graph_to_run_opt = GraphToRunWithOpt()
    x = flow.Tensor(np.random.randn(4, 2, 8, 3)).to("mlu")

    eager_res = flow.relu(get_bn(fused=False)(x))
    eager_res_fuse = get_bn()(x)
    lazy_res = graph_to_run(x)
    lazy_res_opt = graph_to_run_opt(x)
    test_case.assertTrue(np.array_equal(eager_res.numpy(), eager_res_fuse.numpy()))
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res.numpy()))
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res_opt.numpy()))


def do_normalization_add_relu_graph(test_case):
    def get_bn(fused=True):
        if fused:
            bn = flow.nn.FusedBatchNorm2d(num_features=2, eps=1e-5, momentum=0.1).to(
                "mlu"
            )
        else:
            bn = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1).to("mlu")
        bn.eval()
        return bn

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = get_bn()
            self.m.eval()

        def build(self, x, addend):
            return self.m(x, addend=addend)

    class GraphToRunWithOpt(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = get_bn(fused=False)
            self.m.eval()

        def build(self, x, addend):
            return flow.relu(self.m(x) + addend)

    graph_to_run = GraphToRun()
    graph_to_run_opt = GraphToRunWithOpt()
    x = flow.Tensor(np.random.randn(4, 2, 8, 3)).to("mlu")
    addend = flow.Tensor(np.random.randn(4, 2, 8, 3)).to("mlu")

    eager_res = flow.relu(get_bn(fused=False)(x) + addend)
    eager_res_fuse = get_bn()(x, addend=addend)
    lazy_res = graph_to_run(x, addend)
    lazy_res_opt = graph_to_run_opt(x, addend)
    test_case.assertTrue(np.array_equal(eager_res.numpy(), eager_res_fuse.numpy()))
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res.numpy()))
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res_opt.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestNormalizationRelu(oneflow.unittest.TestCase):
    def test_normalization_add_relu_graph(test_case):
        do_normalization_relu_graph(test_case)
        do_normalization_add_relu_graph(test_case)


if __name__ == "__main__":
    unittest.main()
