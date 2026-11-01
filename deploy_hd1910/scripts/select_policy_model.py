"""Select a compatible ONNX while preserving this robot's hardware calibration."""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from microduck_deploy.config import load_config, save_config
from microduck_deploy.policy import OnnxPolicy


def select_model(model, seconds, max_vx, *, root=ROOT):
    seconds, max_vx = float(seconds), float(max_vx)
    if not math.isfinite(seconds) or seconds < 0 or 0 < seconds < 3:
        raise ValueError('时长须为至少3秒或0（持续运行直到停止）')
    if not math.isfinite(max_vx) or not 0 < max_vx <= .4:
        raise ValueError('手柄最大前后速度须为大于0、不超过0.4 m/s')
    if not isinstance(model, str) or not model.strip() or any(char in model for char in '\0\n\r'):
        raise ValueError('模型路径不能为空或包含换行/空字符')
    path = Path(model)
    if path.suffix.lower() != '.onnx':
        raise ValueError('模型路径须为.onnx文件')
    root = Path(root)
    resolved = path if path.is_absolute() else root/path
    # Validate shape, HOME, joint order, observation layout and action scale
    # before making any configuration change. This sends no servo commands.
    OnnxPolicy(resolved)
    cfg = load_config(root/'robot.json')
    if cfg.get('model') != str(path):
        cfg['model'] = str(path)
        save_config(root/'robot.json', cfg)
    print(f'已选择模型：{path}；沿用当前舵机、HOME、IMU和控制配置。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('seconds', type=float)
    parser.add_argument('max_vx', type=float)
    args = parser.parse_args()
    select_model(args.model, args.seconds, args.max_vx)


if __name__ == '__main__':
    main()
