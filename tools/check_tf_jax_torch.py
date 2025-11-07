#!/usr/bin/env python3

def tf_has_gpu() -> bool:
    try:
        import tensorflow as tf  # noqa: F401
        # Avoid deprecated APIs that trigger warnings
        return len(tf.config.list_physical_devices('GPU')) > 0
    except Exception:
        return False


def jax_has_gpu() -> bool:
    try:
        import jax  # noqa: F401
        # Listing devices is sufficient and avoids running computations that
        # can fail when cuDNN/cuBLAS are not fully configured.
        return len(jax.devices('gpu')) > 0
    except Exception:
        return False


def torch_has_gpu() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    print("tf on gpu is available? {}".format(tf_has_gpu()))
    print("jax on gpu is available? {}".format(jax_has_gpu()))
    print("torch on gpu is available? {}".format(torch_has_gpu()))
