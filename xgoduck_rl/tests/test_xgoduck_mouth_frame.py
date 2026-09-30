"""XGODUCK mouth_tip frame matches MicroDuck: +X fwd, +Y left, +Z up."""

import mujoco
import numpy as np

from pathlib import Path

_ROBOT = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/xgoduck"
XGODUCK_WALK_XML = _ROBOT / "robot_walk.xml"
XGODUCK_ALLCOLLISIONS_XML = _ROBOT / "robot_allcollisions.xml"


def _mouth_axes(xml):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # INIT: xml default joints 0
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    assert sid >= 0
    return data.site_xmat[sid].reshape(3, 3).copy()


def test_xgoduck_mouth_tip_matches_microduck_axes_at_cad_zero():
    for xml in (XGODUCK_WALK_XML, XGODUCK_ALLCOLLISIONS_XML):
        R = _mouth_axes(xml)
        np.testing.assert_allclose(R[:, 0], [1, 0, 0], atol=1e-3)  # +X forward
        np.testing.assert_allclose(R[:, 1], [0, 1, 0], atol=1e-3)  # +Y left
        np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-3)  # +Z up
