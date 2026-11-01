"""Validate and preload all selectable sit/stand policies before device IO."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

from .policy import HOME, OnnxPolicy, build_obs61

SITSTAND_KEYS = frozenset('ghimn')


def sitstand_options(cfg):
    options = cfg.get('keyboard_sitstand_models')
    if options is None:
        return {}
    if not isinstance(options, dict) or set(options) != SITSTAND_KEYS:
        raise ValueError('keyboard_sitstand_models 必须配置 g/h/i/m/n 五个坐站模型')
    for key, item in options.items():
        if not isinstance(item, dict):
            raise ValueError(f'坐站模型 {key} 配置无效')
        path, label, digest = (item.get(name) for name in ('model', 'label', 'sha256'))
        if (not isinstance(path, str) or not path.strip()
                or Path(path).suffix.lower() != '.onnx'
                or not isinstance(label, str) or not label.strip()
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError(f'坐站模型 {key} 需要 ONNX 路径、名称和 SHA256')
    default = cfg.get('keyboard_sitstand_default')
    if not isinstance(default, str) or default not in options:
        raise ValueError('keyboard_sitstand_default 必须是已配置的坐站字母键')
    if options[default]['model'] != cfg.get('sitstand_model'):
        raise ValueError('默认坐站字母必须与 sitstand_model 一致')
    return options


def load_sitstand_policies(cfg, root, *, base_policy=None, policy_loader=OnnxPolicy):
    options = sitstand_options(cfg)
    for key, item in options.items():
        path = Path(root) / item['model']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'坐站模型 {key} 校验失败：{path}')
    sessions = {}
    for key, item in options.items():
        session = (base_policy if key == cfg['keyboard_sitstand_default'] and base_policy is not None
                   else policy_loader(Path(root) / item['model'], policy_kind='sitstand'))
        for sit_flag in (0., 1.):
            session.infer(build_obs61(np.zeros(3), [0, 0, -1], HOME,
                                     np.zeros(14), np.zeros(14), twist=(sit_flag, 0., 0.)))
        sessions[key] = session
    return sessions
