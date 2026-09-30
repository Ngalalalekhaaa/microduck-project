"""XGODUCK collisions: every link vs ground; only head_roll vs base self-collides."""

import mujoco

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_ALLCOLLISIONS_XML,
    XGODUCK_COLLISION,
    XGODUCK_WALK_XML,
)

# Standup/allcollisions: every link has a named *_collision mesh for floor
# contact. Bit 0 = world/floor, bit 1 = head-base self-collision.
STANDUP_WORLD_COLLISION_GEOMS = {
    "trunk_collision": "trunk_base",
    "left_hip_yaw_collision": "yaw2roll",
    "left_hip_roll_collision": "hip_l",
    "left_thigh_collision": "upper_leg_left",
    "left_shin_collision": "leg",
    "left_foot_collision": "ankle_left",
    "neck_collision": "neck",
    "head_pitch_collision": "neck_pitch",
    "head_yaw_collision": "yaw_roll_motion",
    "head_collision": "jaw_soft",
    "right_hip_yaw_collision": "bearing_roll",
    "right_hip_roll_collision": "hip_l_2",
    "right_thigh_collision": "upper_leg_right",
    "right_shin_collision": "leg_2",
    "right_foot_collision": "ankle_right",
}
SELF_COLLISION_PAIR = frozenset({"trunk_collision", "head_collision"})
SELF_COLLISION_GEOMS = frozenset(SELF_COLLISION_PAIR)


def _compile_with_collision(xml_path, collision_cfg):
    spec = mujoco.MjSpec.from_file(str(xml_path))
    collision_cfg.edit_spec(spec)
    # Match Entity.spec_fn: unnamed geoms must not leak world contacts.
    for geom in spec.geoms:
        if not geom.name:
            geom.contype = 0
            geom.conaffinity = 0
    return spec.compile()


def _pair_collides(model, i, j) -> bool:
    return bool(
        (int(model.geom_contype[i]) & int(model.geom_conaffinity[j]))
        or (int(model.geom_contype[j]) & int(model.geom_conaffinity[i]))
    )


def test_xgoduck_standup_all_links_hit_ground_only_head_and_base_self_collide():
    model = _compile_with_collision(XGODUCK_ALLCOLLISIONS_XML, XGODUCK_COLLISION)

    active = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        ct = int(model.geom_contype[i])
        ca = int(model.geom_conaffinity[i])
        if ct or ca:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
            assert name, f"unnamed active geom on body {body!r}"
            assert name in STANDUP_WORLD_COLLISION_GEOMS, (
                f"unexpected active geom {name!r} on body {body!r}"
            )
            active[name] = (i, body, ct, ca)

    missing = set(STANDUP_WORLD_COLLISION_GEOMS) - set(active)
    assert not missing, f"missing world-collision geoms: {sorted(missing)}"

    for name, expected_body in STANDUP_WORLD_COLLISION_GEOMS.items():
        _, body, ct, ca = active[name]
        assert body == expected_body, f"{name} on {body!r}, expected {expected_body!r}"
        if name in SELF_COLLISION_GEOMS:
            # Floor (bit 0) + head↔base (bit 1).
            assert ct == 3 and ca == 2, f"{name} bits ct={ct} ca={ca}"
        else:
            # Floor only; must not participate in robot-robot contacts.
            assert ct == 1 and ca == 0, f"{name} bits ct={ct} ca={ca}"

    saw_self = False
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j) or ""
            collides = _pair_collides(model, i, j)
            if frozenset({n1, n2}) == SELF_COLLISION_PAIR:
                assert collides, "head_roll and base must self-collide"
                saw_self = True
            else:
                assert not collides, f"unexpected self-collision {n1!r} vs {n2!r}"
    assert saw_self


def test_xgoduck_walk_has_no_thigh_world_collision():
    model = _compile_with_collision(XGODUCK_WALK_XML, XGODUCK_COLLISION)
    for name in ("left_thigh_collision", "right_thigh_collision", "trunk_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid < 0


def test_xgoduck_standup_task_enables_head_base_self_collision():
    cfg = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    col = cfg.scene.entities["robot"].collisions[0]
    assert col.contype["^(trunk_collision|head_collision)$"] == 3
    assert col.conaffinity["^(trunk_collision|head_collision)$"] == 2
