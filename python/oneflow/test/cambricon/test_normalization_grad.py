import itertools
import numpy as np
import oneflow as flow


def _test_batchnorm_grad(shape, track_running_stats):
    np_x = np.random.randn(*shape).astype(np.float32)
    cpu_x = flow.tensor(np_x).to("cpu").requires_grad_(True)
    mlu_x = flow.tensor(np_x).to("mlu").requires_grad_(True)
    cpu_batch_norm = (
        flow.nn.BatchNorm2d(
            num_features=int(cpu_x.shape[1]),
            track_running_stats=track_running_stats,
            affine=True,
        )
        .train()
        .to("cpu")
    )
    mlu_batch_norm = (
        flow.nn.BatchNorm2d(
            num_features=int(mlu_x.shape[1]),
            track_running_stats=track_running_stats,
            affine=True,
        )
        .train()
        .to("mlu")
    )
    cpu_batch_norm(cpu_x).sum().backward()
    mlu_batch_norm(mlu_x).sum().backward()

    assert np.allclose(
        cpu_batch_norm.weight.grad.numpy(),
        mlu_batch_norm.weight.grad.numpy(),
        1e-4,
        1e-4,
    )
    assert np.allclose(
        cpu_batch_norm.bias.grad.numpy(), mlu_batch_norm.bias.grad.numpy(), 1e-4, 1e-4
    )
    assert np.allclose(cpu_x.grad.numpy(), mlu_x.grad.numpy(), 1e-4, 1e-4)


def test_batchnorm_grad():
    for shape, track_running_stats in itertools.product(
        [(2, 3, 4, 5), (1, 2, 3, 4), (5, 6, 7, 8)], [True, False]
    ):
        _test_batchnorm_grad(shape, track_running_stats)
