"""Validate and preload opt-in keyboard walking sessions before hardware IO."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

from .policy import HOME, OnnxPolicy, build_obs61


def walking_options(cfg):
    options = cfg.get('keyboard_walking_models')
    if options is None:
        return {}
    if not isinstance(options, dict) or set(options) != set('abcdef'):
        raise ValueError('keyboard_walking_models 必须配置 a–f 六个行走模型')
    for key, item in options.items():
        if not isinstance(item, dict):
            raise ValueError(f'行走模型 {key} 配置无效')
        path, label, digest = (item.get(name) for name in ('model', 'label', 'sha256'))
        if (not isinstance(path, str) or not path.strip()
                or Path(path).suffix.lower() != '.onnx'
                or not isinstance(label, str) or not label.strip()
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError(f'行走模型 {key} 需要 ONNX 路径、名称和 SHA256')
    if options['a']['model'] != cfg['model']:
        raise ValueError('默认行走模型必须与按键 a 一致')
    return options


def load_walking_policies(cfg, root, *, base_policy=None, policy_loader=OnnxPolicy):
    options = walking_options(cfg)
    # Check the entire set before creating sessions; never load a model in-loop.
    for key, item in options.items():
        path = Path(root) / item['model']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'行走模型 {key} 校验失败：{path}')
    sessions = {}
    for key, item in options.items():
        session = (base_policy if key == 'a' and base_policy is not None else
                   policy_loader(Path(root) / item['model'], policy_kind='walking'))
        for twist in ((0., 0., 0.), (.3, 0., 0.), (0., 0., .5)):
            session.infer(build_obs61(np.zeros(3), [0, 0, -1], HOME,
                                     np.zeros(14), np.zeros(14), twist=twist))
        sessions[key] = session
    return sessions
