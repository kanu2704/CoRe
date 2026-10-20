"""Small helpers so CoRe's code runs on a fresh Python without Hydra 0.11 or
CoRe's runtime edits to the MetaWorld source files."""
import importlib
import sys
import types

import numpy as np


def instantiate(cfg, *args, **kwargs):
    """Minimal stand-in for hydra 0.11's utils.instantiate({'class': ..., 'params': {...}})."""
    module_name, cls_name = cfg['class'].rsplit('.', 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = dict(cfg.get('params', {}))
    params.update(kwargs)
    return cls(*args, **params)


def install_hydra_shim():
    """agent/sac.py only uses hydra.utils.instantiate; provide it without Hydra."""
    hydra = types.ModuleType('hydra')
    utils = types.ModuleType('hydra.utils')
    utils.instantiate = instantiate
    hydra.utils = utils
    sys.modules['hydra'] = hydra
    sys.modules['hydra.utils'] = utils


def base_env(env):
    """Walk through gym / rlkit wrappers down to the MetaWorld env."""
    e = env
    for _ in range(10):
        if hasattr(e, '_target_pos'):
            return e
        e = getattr(e, 'env', None) or getattr(e, '_wrapped_env', None)
        if e is None:
            break
    raise AttributeError("could not find the MetaWorld env inside the wrappers")


def target_pos(env):
    """Same value CoRe stores as env.target_pos_frm at reset (env._target_pos)."""
    return np.array(base_env(env)._target_pos, dtype=np.float64).copy()


def reset_env(env):
    obs = env.reset()
    return obs[0] if isinstance(obs, tuple) else obs


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    return out


def gl_renderer():
    """Name of the OpenGL renderer of the current context, e.g. 'Tesla T4/PCIe/SSE2' (GPU)
    or 'llvmpipe ...' (software). Call right after env.render(); None if unknown."""
    try:
        from OpenGL import GL
        name = GL.glGetString(GL.GL_RENDERER)
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None


SCRIPTED_POLICIES = {
    'peg-insert-side-v2': 'SawyerPegInsertionSideV2Policy',
    'hammer-v2': 'SawyerHammerV2Policy',
    'button-press-topdown-v2': 'SawyerButtonPressTopdownV2Policy',
    'dial-turn-v2': 'SawyerDialTurnV2Policy',
    'drawer-open-v2': 'SawyerDrawerOpenV2Policy',
    'soccer-v2': 'SawyerSoccerV2Policy',
    'sweep-into-v2': 'SawyerSweepIntoV2Policy',
}


def scripted_policy(task):
    policies = importlib.import_module('metaworld.policies')
    return getattr(policies, SCRIPTED_POLICIES[task])()
