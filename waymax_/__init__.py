
# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Compatibility shim for older WAYMAX dependencies expecting the legacy
# ShardedDeviceArray type from JAX. Recent JAX versions removed this symbol, so
# we alias it to the modern jax.Array when missing.
import sys
from types import ModuleType

try:
    import jax
    from jax.interpreters import pxla as _pxla  # type: ignore
    from jax.interpreters import xla as _xla  # type: ignore
    from jax import random as _random  # type: ignore
    from jax import core as _core  # type: ignore
    from jax import tree_util as _tree_util  # type: ignore
    from jax._src import config as _config_mod  # type: ignore
    from jax._src import linear_util as _linear_util  # type: ignore
    import jax.numpy as _jnp  # type: ignore
    import jax.nn as _nn  # type: ignore

    if hasattr(jax, "Array"):
        if not hasattr(_pxla, "ShardedDeviceArray"):
            _pxla.ShardedDeviceArray = jax.Array  # type: ignore[attr-defined]
        if not hasattr(_xla, "DeviceArray"):
            _xla.DeviceArray = jax.Array  # type: ignore[attr-defined]
        if not hasattr(_random, "KeyArray"):
            _random.KeyArray = jax.Array  # type: ignore[attr-defined]
        if not hasattr(_core, "Shape"):
            _core.Shape = tuple  # type: ignore[attr-defined]
        if not hasattr(_tree_util, "register_keypaths"):
            def _register_keypaths(clz, paths):  # pragma: no cover - shim
                return None
            _tree_util.register_keypaths = _register_keypaths  # type: ignore[attr-defined]
        if not hasattr(jax, "linear_util"):
            jax.linear_util = _linear_util  # type: ignore[attr-defined]
        if "jax.config" not in sys.modules:
            config_module = ModuleType("jax.config")
            for _name in dir(_config_mod):
                if _name.startswith("_"):
                    continue
                setattr(config_module, _name, getattr(_config_mod, _name))
            sys.modules["jax.config"] = config_module
        if not hasattr(jax, "config"):
            jax.config = _config_mod  # type: ignore[attr-defined]
        if not hasattr(_nn, "normalize"):
            def _normalize(x, axis=-1, epsilon=1e-12):
                denom = _jnp.linalg.norm(x, axis=axis, keepdims=True)
                denom = _jnp.maximum(denom, epsilon)
                return x / denom
            _nn.normalize = _normalize  # type: ignore[attr-defined]
        if not hasattr(jax, "ShapedArray"):
            jax.ShapedArray = _core.ShapedArray  # type: ignore[attr-defined]
except Exception:
    pass
