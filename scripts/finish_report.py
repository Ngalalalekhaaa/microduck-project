"""Create a local result report and learning curves after the training queue exits."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
run_dir = args.run.resolve()
state = json.loads((run_dir / 'status.json').read_text())
root = Path(__file__).resolve().parents[1]
robot = state['args'].get('robot', 'xl330')
description = ('HD-1910 公开 BAM M6 参数基线（尚未按本机辨识）'
               if robot == 'hd1910' else '官方 XL330 BAM M6 模型')
lines = [
    '# Microduck 训练结果', '',
    f'报告生成时间：{datetime.now().astimezone().isoformat()}', '',
    f'流水线状态：**{state["status"]}**。', '',
    f'使用{description}，两版独立随机初始化；每版目标 {state["args"]["iterations"]} 轮、'
    f'{state["args"]["num_envs"]} 个环境，种子 42。',
    '齿隙版为每个驱动关节增加 ±1° 的被动机械齿隙。', '',
    '评估每条命令 64 个环境、10 秒，固定头部和身体命令为零，关闭外部推力，',
    '保留上游启动/重置随机化。存活率按首回合统计；速度误差只统计第 1 秒后、',
    '首次失败前的样本，需结合存活率一起看。本报告不代表实机验证。', '',
    '| 训练模型 | 测试环境 | 命令 vx,vy,wz | 10 秒存活率 | 实际平均 vx,vy,wz | 速度绝对误差 vx,vy,wz |',
    '|---|---|---|---:|---|---|',
]


def vec(value):
    return ', '.join(f'{x:.3f}' for x in value) if value is not None else '无有效样本'


for item in state['runs']:
    variant = item['variant']
    artifacts = Path(item.get('artifacts', run_dir / variant))
    for evaluation in sorted(artifacts.glob('eval_*/evaluation.json')):
        result = json.loads(evaluation.read_text())
        for name, metrics in result['results'].items():
            lines.append(f'| {variant} | {evaluation.parent.name} | {name}: {vec(metrics["command_vx_vy_wz"])} | '
                         f'{metrics["survival_fraction"]:.1%} | {vec(metrics["mean_actual_vx_vy_wz"])} | '
                         f'{vec(metrics["mean_abs_error_vx_vy_wz"])} |')
lines += ['', '## 模型与原始证据', '']
for item in state['runs']:
    lines += [f'### {item["variant"]}', '', f'- 任务：`{item["task"]}`',
              f'- 状态：{item["status"]}', f'- 训练日志：[{Path(item["stdout"]).name}]({item["stdout"]})']
    if item.get('checkpoint'):
        lines.append(f'- 检查点：[{Path(item["checkpoint"]).name}]({item["checkpoint"]})')
    if item.get('artifacts'):
        artifact = Path(item['artifacts'])
        for path in [artifact / 'policy.onnx', *sorted(artifact.glob('eval_*/evaluation.json')),
                     *sorted(artifact.glob('eval_*/forward.mp4'))]:
            if path.exists():
                lines.append(f'- [{path.relative_to(artifact)}]({path})')
    lines.append('')

# Read TensorBoard files without importing or allocating a simulation on GPU.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for item in state['runs']:
    for directory in item.get('training_dirs', []):
        acc = EventAccumulator(directory, size_guidance={'scalars': 0})
        acc.Reload()
        tags = acc.Tags().get('scalars', [])
        for axis, ending, title in zip(axes, ['mean_reward', 'mean_episode_length'],
                                       ['Mean training reward', 'Mean episode length (steps)']):
            candidates = [tag for tag in tags if tag.lower().endswith(ending)]
            if candidates:
                entries = acc.Scalars(candidates[0])
                axis.plot([e.step for e in entries], [e.value for e in entries],
                          label=item['variant'], linewidth=1.2)
            axis.set(xlabel='PPO iteration', title=title)
            axis.grid(alpha=0.25)
for axis in axes:
    if axis.lines:
        axis.legend()
fig.tight_layout()
plot = run_dir / 'training_curves.png'
fig.savefig(plot, dpi=160)
plt.close(fig)
lines += ['## 步态判断', '',
          '训练完成和短时不摔倒不代表能够跟踪行走指令，应同时检查实际速度与视频。', '']
for item in state['runs']:
    evaluation = Path(item.get('artifacts', run_dir / item['variant'])) / f'eval_{item["variant"]}/evaluation.json'
    if evaluation.exists():
        metrics = json.loads(evaluation.read_text())['results'].get('forward')
        if metrics and metrics['mean_actual_vx_vy_wz'] is not None:
            actual = metrics['mean_actual_vx_vy_wz'][0]
            target = metrics['command_vx_vy_wz'][0]
            judgement = '低速前进跟踪明显不足。' if actual < target * 0.5 else '仍需结合其他指令与视频判断。'
            lines.append(f'- {item["variant"]}：目标前进 {target:.3f} m/s，实际均值 {actual:.4f} m/s；{judgement}')
lines += ['', '## 训练曲线', '', f'![训练曲线]({plot})', '',
          ('HD-1910 动力学采用公开参数；质量惯量、齿隙、电压范围及控制增益仍需与实机校准。'
           if robot == 'hd1910' else '当前 XL330 策略不能视为用户 HD-1910 真机的适配版本。'),
          '本报告仅包含仿真验证。', '']
report = args.output or root / 'reports/训练结果.md'
report.write_text('\n'.join(lines))
print(report)
