import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype")

import math
import requests

import hydra
import jax
try:
    from jax.interpreters import pxla as _pxla  # type: ignore
    if hasattr(jax, "Array") and not hasattr(_pxla, "ShardedDeviceArray"):
        _pxla.ShardedDeviceArray = jax.Array  # type: ignore[attr-defined]
except Exception:
    pass
import src.utils.init_default_jax  # noqa: F401
import torch
from omegaconf import OmegaConf

from simulator.engines.ltd_simulator import LTDSimulator
from src.policy import build_model
from src.utils.utils import update_waymax_config
@hydra.main(version_base=None, config_path="configs", config_name="simulate")
def simulate(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    if not hasattr(cfg, "batch_dims") or cfg.batch_dims is None:
        raise ValueError("`batch_dims` must be provided for simulation.")

    batch_dims = list(cfg.batch_dims)
    if len(batch_dims) < 2:
        raise ValueError(f"`batch_dims` should have at least 2 dimensions, got {batch_dims}.")

    local_devices = jax.local_device_count()
    if local_devices < 1:
        raise RuntimeError("No JAX devices available. Check your JAX installation.")

    requested_devices = int(batch_dims[0])
    if requested_devices > local_devices:
        print(
            f"Adjusting batch_dims first axis from {requested_devices} to available device count "
            f"{local_devices}.",
        )
        scale = requested_devices / local_devices
        current_per_device = int(batch_dims[1])
        adjusted_per_device = max(1, math.ceil(current_per_device / scale))
        if adjusted_per_device != current_per_device:
            print(
                f"Reducing per-device batch from {current_per_device} to {adjusted_per_device} "
                "to fit within available device memory.",
            )
        batch_dims[0] = local_devices
        batch_dims[1] = adjusted_per_device
        cfg.batch_dims = batch_dims

    model = build_model(cfg)
    cfg = update_waymax_config(cfg)
    if cfg.ckpt_path is not None:
        model.load_state_dict(torch.load(cfg.ckpt_path,map_location='cuda:{}'.format(torch.cuda.device_count()-1))['state_dict'])
        print(f'Loaded {cfg.ckpt_path}')
    else:
        print('No ckpt provided')
    runner = LTDSimulator(
        model = model,
        config=cfg,
        batch_dims=cfg.batch_dims,
    )
    eval_rtg = 0
    ep_return = [eval_rtg] * runner.env.num_envs
    runner.run(ep_return=ep_return, vis =cfg.vis)
if __name__ == '__main__':
    message="Something Went Wrong! (code seems not run)".encode(encoding='utf-8')
    try:
        simulate()
        message="Simulation done successful".encode(encoding='utf-8')
    except Exception as e:
        message=f"Simulation failed with error: {e}".encode(encoding='utf-8')
        raise e
    finally:
        requests.post("https://ntfy.sh/shnamtopic", data=message)