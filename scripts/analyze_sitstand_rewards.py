"""Measure reward mass on successful original-policy cycles and standing traps.

Read the reward manager's actual per-step terms once, including stateful costs.
Reweight the SAME trajectories offline to separate newly added/strengthened
costs from dynamics changes. No optimization, export or robot access occurs.
"""

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.models import MLPModel
from mjlab_microduck.tasks.microduck_hd1910_sitstand_robust_env_cfg import (
    TASK_ID, ROBUST_STAGES, make_microduck_hd1910_sitstand_robust_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'microduck_rl_hd1910/logs/rsl_rl/microduck_hd1910_sitstand_robust/2026-11-01_03-25-57_sitstand_1699_wide_dr_35deg_20261101_032546'
DT = .02
HORIZON = 650
PHASES = {'whole_cycle': (0., 13.), 'initial_stand': (0., 3.),
          'descent': (3., 5.), 'seated_hold': (5., 8.),
          'rise': (8., 10.), 'standing_hold': (10., 13.)}


def weights_at_original_iteration(iteration=1699):
    cfg = make_microduck_sitstand_env_cfg()
    weights = {n: c.weight for n, c in cfg.rewards.items()}
    for curriculum in cfg.curriculum.values():
        params = curriculum.params
        if 'reward_name' in params and 'weight_stages' in params:
            for stage in params['weight_stages']:
                if stage['step'] <= iteration * 24:
                    weights[params['reward_name']] = stage['weight']
    return weights


def make_cfg(profile, count, seed):
    cfg = make_microduck_hd1910_sitstand_robust_env_cfg(
        evaluation_profile={'nominal': 'nominal', 'initial_dr': 'train', 'full_dr': 'stress'}[profile])
    cfg.curriculum = {}
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = count, seed, False
    cfg.episode_length_s = 15.
    cfg.commands['twist'].resampling_time_range = (1000., 1000.)
    cfg.commands['head_pose'].ranges = ((0., 0.),) * 4
    cfg.commands['head_pose'].zero_command_prob = 1.
    cfg.events['set_ground_state'].params.update({'sitting_prob': 0., 'standing_prob': 1.})
    cfg.events['reset_base'].params['pose_range'].update({'roll': (-.015, .015), 'pitch': (-.015, .015)})
    cfg.events['reset_robot_joints'].params['position_range'] = (-.015, .015)
    if profile == 'nominal':
        cfg.events['sitstand_response'].params['ankle_kp_range'] = (1., 1.)
    for name, weight in ROBUST_STAGES[-1]['reward_weights'].items():
        cfg.rewards[name].weight = weight
    return cfg


def build_actor(obs, checkpoint, device):
    agent = asdict(load_rl_cfg(TASK_ID))
    kwargs = {k: deepcopy(v) for k, v in agent['actor'].items()
              if k in inspect.signature(MLPModel.__init__).parameters}
    actor = MLPModel(obs, agent['obs_groups'], 'actor', 14, **kwargs).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(state['actor_state_dict'])
    return actor


def summarize(raw, active, selected, terms, weight_sets):
    selected = np.asarray(selected, dtype=bool)
    if not selected.any():
        return {'worlds': 0, 'phases': {}}
    output = {'worlds': int(selected.sum()), 'phases': {}}
    time = np.arange(HORIZON) * DT
    for phase, (lo, hi) in PHASES.items():
        mask = active & selected[None, :] & ((time >= lo) & (time < hi))[:, None]
        if not mask.any():
            continue
        raw_mean = raw[mask].mean(axis=0)
        summary = {'frames': int(mask.sum()), 'raw_mean': dict(zip(terms, raw_mean.tolist())), 'scores': {}}
        for label, weights in weight_sets.items():
            w = np.array([weights.get(n, 0.) for n in terms])
            rates = raw_mean * w
            reward = np.sum(raw * w, axis=2)
            total_by_world = (np.where(mask, reward, 0.) * DT).sum(axis=0)
            gamma = .99 ** np.maximum(0, np.arange(HORIZON) - round(lo / DT))
            discounted = (np.where(mask, reward, 0.) * DT * gamma[:, None]).sum(axis=0)
            summary['scores'][label] = {
                'weighted_mean_rate': dict(zip(terms, rates.tolist())),
                'weighted_mean_per_step': dict(zip(terms, (rates * DT).tolist())),
                'positive_mean_rate': float(np.maximum(rates, 0).sum()),
                'negative_mean_rate': float(np.minimum(rates, 0).sum()),
                'net_mean_rate': float(rates.sum()),
                'mean_total_per_world': float(total_by_world[selected].mean()),
                'mean_discounted_total_per_world': float(discounted[selected].mean()),
            }
        output['phases'][phase] = summary
    # Discounted advantage from the SIT command, including subsequent standing.
    mask = active & selected[None, :] & (time >= 3.)[:, None]
    output['discounted_return_from_sit_command'] = {}
    for label, weights in weight_sets.items():
        w = np.array([weights.get(n, 0.) for n in terms])
        reward = (raw * w).sum(axis=2)
        gamma = .99 ** np.maximum(0, np.arange(HORIZON) - 150)
        values = (np.where(mask, reward, 0.) * DT * gamma[:, None]).sum(axis=0)
        output['discounted_return_from_sit_command'][label] = float(values[selected].mean())
    return output


@torch.inference_mode()
def rollout(env, label, checkpoint, stochastic, ignore_sit, seed, output):
    obs, _ = env.reset(seed=seed)
    actor = build_actor(TensorDict(obs, batch_size=[env.num_envs]), checkpoint, env.device)
    command = env.command_manager.get_term('twist')
    names = env.reward_manager.active_terms
    weights = {n: env.reward_manager.get_term_cfg(n).weight for n in names}
    if any(v == 0 for v in weights.values()):
        raise RuntimeError('All analyzed terms must be active to recover raw values')
    weight_tensor = torch.tensor([weights[n] for n in names], device=env.device)
    count = env.num_envs
    raw = torch.zeros(HORIZON, count, len(names), device=env.device)
    active = torch.zeros(HORIZON, count, dtype=torch.bool, device=env.device)
    heights = torch.zeros(HORIZON, count, device=env.device)
    tilts = torch.zeros_like(heights)
    flags = torch.zeros_like(heights)
    alive = torch.ones(count, dtype=torch.bool, device=env.device)
    failures = []
    for step in range(HORIZON):
        t = step * DT
        sit = float(3. <= t < 8.)
        command.vel_command_b.zero_()
        command.vel_command_b[:, 0] = sit
        obs = env.observation_manager.compute(update_history=False)
        actor_obs = obs['actor'].clone()
        actor_obs[:, 48:51] = command.vel_command_b
        if ignore_sit:
            actor_obs[:, 48] = 0.
        action = actor(TensorDict({'actor': actor_obs}, batch_size=[count]), stochastic_output=stochastic)
        active[step] = alive
        _, reward, terminated, timed_out, _ = env.step(action)
        weighted = env.reward_manager._step_reward.clone()
        torch.testing.assert_close(weighted.sum(dim=1) * DT, reward, rtol=2.e-5, atol=2.e-6)
        raw[step] = weighted / weight_tensor
        robot = env.scene['robot']
        heights[step] = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
        tilts[step] = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.)))
        flags[step] = sit
        for i in ((terminated | timed_out) & alive).nonzero(as_tuple=False).flatten().tolist():
            reasons = [n for n in env.termination_manager.active_terms
                       if bool(env.termination_manager.get_term(n)[i])]
            failures.append({'env': i, 'time_s': round((step+1)*DT, 3), 'reasons': reasons,
                             'tilt_deg': float(tilts[step, i])})
        alive &= ~(terminated | timed_out)
        reset = (terminated | timed_out).nonzero(as_tuple=False).flatten()
        if len(reset):
            env.reset(env_ids=reset)
    raw, active, heights, tilts, flags = [x.cpu().numpy() for x in (raw, active, heights, tilts, flags)]
    success = alive.cpu().numpy().copy()
    time = np.arange(HORIZON) * DT
    window_stats = {}
    for name, lo, hi, ideal in [('stand', 2, 3, .115), ('sit', 7, 8, .060), ('returned', 12, 13, .115)]:
        mask = (time >= lo) & (time < hi)
        z = heights[mask].mean(axis=0)
        reached = active[mask].all(axis=0) & (np.abs(z - ideal) < .020)
        success &= reached
        window_stats[name] = {'height_success_worlds': int(reached.sum()),
                              'surviving_mean_height_m': float(z[alive.cpu().numpy()].mean()) if alive.any() else None}
    # Termination uses derived physics data one substep old. Require the
    # recorded post-step peak to satisfy the same 35-degree bound as well.
    success &= tilts.max(axis=0) <= 35.
    baseline = weights_at_original_iteration()
    initial = dict(weights, **ROBUST_STAGES[0]['reward_weights'])
    weight_sets = {'original_recipe_1699': baseline, 'continuation_initial': initial, 'continuation_final': weights}
    result = {'label': label, 'checkpoint': str(checkpoint), 'stochastic': stochastic,
              'ignore_sit_in_actor_only': ignore_sit, 'seed': seed, 'worlds': count,
              'successful_cycles': int(success.sum()), 'survived': int(alive.sum()),
              'failures': failures, 'windows': window_stats,
              'max_tilt_deg_successful': float(tilts[:, success].max()) if success.any() else None,
              'reward_names': names, 'weight_sets': weight_sets,
              'successful': summarize(raw, active, success, names, weight_sets),
              'all_first_episodes': summarize(raw, active, np.ones(count, bool), names, weight_sets)}
    stem = output / label
    np.savez_compressed(stem.with_suffix('.npz'), raw=raw, active=active, success=success,
                        heights=heights, tilts=tilts, flags=flags, names=np.array(names))
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--profiles', nargs='+', default=['nominal', 'initial_dr', 'full_dr'])
    parser.add_argument('--policies', nargs='+', default=['original', 'original_exploration', 'ignore_sit', 'final4999'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'control_dt_s': DT, 'gamma': .99, 'horizon_s': 13., 'results': [],
              'units': 'raw_value * weight is reward rate; per-control-step reward is rate * 0.02',
              'scope': 'Simulation only; successful conditional means and all first episodes are both reported.',
              'baseline_recipe_scope': 'Current repository sitstand weights at iteration 1699; original external ONNX training configuration is unavailable.'}
    definitions = {'original': ('warm_start.pt', False, False),
                   'original_exploration': ('warm_start.pt', True, False),
                   'ignore_sit': ('warm_start.pt', False, True),
                   'final4999': ('model_4999.pt', False, False)}
    for profile in args.profiles:
        with torch.inference_mode():
            env = ManagerBasedRlEnv(make_cfg(profile, args.num_envs, 42), device='cuda:0')
            try:
                for policy in args.policies:
                    checkpoint, stochastic, ignore = definitions[policy]
                    label = f'{profile}__{policy}'
                    result = rollout(env, label, RUN/checkpoint, stochastic, ignore, 42, args.output)
                    report['results'].append(result)
                    (args.output/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
                    print(json.dumps({'profile':profile,'policy':policy,'success':result['successful_cycles'],
                          'worlds':args.num_envs,'survived':result['survived'],
                          'reward_rate':result['all_first_episodes']['phases']['whole_cycle']['scores']['continuation_final']['net_mean_rate']}),flush=True)
            finally:
                env.close()
    rows = []
    for result in report['results']:
        for membership in ('successful', 'all_first_episodes'):
            for phase, statistics in result[membership]['phases'].items():
                for recipe, scores in statistics['scores'].items():
                    for name, value in scores['weighted_mean_rate'].items():
                        rows.append({'rollout':result['label'],'membership':membership,
                                     'phase':phase,'recipe':recipe,'term':name,
                                     'weight':result['weight_sets'][recipe].get(name,0.),
                                     'raw_mean':statistics['raw_mean'][name],
                                     'weighted_mean_rate':value,'weighted_mean_per_step':value*DT})
    with (args.output/'reward_terms.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    main()
