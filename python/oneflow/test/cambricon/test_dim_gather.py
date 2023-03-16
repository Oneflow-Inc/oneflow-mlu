import itertools
import numpy as np
import oneflow as flow


def __test_dim_gather(x_array, index_array, dim, index_dtype):
    x = flow.tensor(x_array, device="cpu", dtype=flow.float32)
    index = flow.tensor(index_array, device="cpu", dtype=index_dtype)
    cpu_out_numpy = flow.gather(x, dim, index).numpy()
    x = x.to("mlu")
    index = index.to("mlu")
    mlu_out_numpy = flow.gather(x, dim, index).numpy()
    assert np.allclose(cpu_out_numpy, mlu_out_numpy, 1e-4, 1e-4)


def test_dim_gather():
    array = (
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 0], [1, 0]]),
    )
    array_multi_dim = (
        np.random.randn(3, 4, 3, 5),
        np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5)),
    )
    arrays_single_dim = zip([np.ones(1), 1.0], [0, 0])
    
    index_dtypes = [flow.int32, flow.int64]

    for dim, dtype in itertools.product([0, 1], index_dtypes):
        __test_dim_gather(*array, dim, dtype)

    for dim, dtype in itertools.product([1, 2, 3], index_dtypes):
        __test_dim_gather(*array_multi_dim, dim, dtype)

    for array_pair, dtype in itertools.product(arrays_single_dim, index_dtypes):
        __test_dim_gather(*array_pair, 0, dtype)
