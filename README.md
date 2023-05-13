# oneflow-mlu

OneFlow-MLU is an OneFlow extension that enables oneflow to run on the Cambricon MLU chips. Currently it only supports the MLU 370 series.

## Installation

### pip

TODO

### Building From Source

#### Prerequisites

- install cmake
- install Cambricon MLU driver (https://sdk.cambricon.com/download?sdk_version=V1.11.0&component_name=Driver )，CNToolKit，CNNL and CNCL (https://sdk.cambricon.com/download?sdk_version=V1.11.0&component_name=Basis ).
- build oneflow with cpu only from source and install it

#### Get the OneFlow-MLU Source

```shell
git clone https://github.com/Oneflow-Inc/oneflow-mlu
```

#### Building

Inside OneFlow-MLU source directory, then run the following command to build and install `oneflow_mlu`,

```shell
python3 setup.py install
```

## Run A Toy Program

```python
# python3

>>> import oneflow as flow
>>> import oneflow_mlu
>>>
>>> m = flow.nn.Linear(3, 4).to("mlu")
>>> x = flow.randn(4, 3, device="mlu")
>>> y = m(x)
>>> print(y)
tensor([[ 0.4239, -0.4689, -0.1660,  0.0718],
        [ 0.5413,  1.9006,  2.0763,  0.8693],
        [ 0.4226, -0.0207,  0.1006,  0.2234],
        [ 0.4054, -0.2816, -0.4405,  0.1099]], device='mlu:0', dtype=oneflow.float32, grad_fn=<broadcast_addBackward>)
```

## Models
- [ResNet50](https://github.com/Oneflow-Inc/oneflow-mlu-models#resnet50)
- [GPT2](https://github.com/Oneflow-Inc/oneflow-mlu-models#libai_gpt2)
