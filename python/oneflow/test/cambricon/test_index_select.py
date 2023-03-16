import numpy as np
import oneflow as flow


def test_index_select():
    shape = np.random.randint(low=1, high=10, size=4).tolist()
    dim = np.random.randint(low=0, high=4, size=1)[0]
    x = flow.tensor(np.random.randn(*shape), device="cpu", dtype=flow.float32)
    index = flow.tensor(
        np.random.randint(
            low=0, high=shape[dim], size=np.random.randint(low=1, high=10, size=1)[0]
        ),
        device="cpu",
        dtype=flow.int32,
    )
    cpu_out_numpy = flow.index_select(x, dim, index).numpy()
    x = x.to("mlu")
    index = index.to("mlu")
    mlu_out_numpy = flow.index_select(x, dim, index).numpy()
    assert np.allclose(cpu_out_numpy, mlu_out_numpy, 1e-4, 1e-4)
