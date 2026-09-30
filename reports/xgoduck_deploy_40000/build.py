from pathlib import Path
import shutil, json, hashlib
root=Path('/home/luckysir/microduck')
source=root/'reports/xgoduck_deploy_40000/source'
out=root/'deploy_xgoduck_40000'
if out.exists(): raise SystemExit('destination already exists')
out.mkdir()
shutil.copytree(source/'microduck_deploy', out/'microduck_deploy', ignore=shutil.ignore_patterns('__pycache__'))
(out/'scripts').mkdir()
for name in ('home_then_policy.py','policy_trial.py','supported_home.py'):
    shutil.copy2(source/'scripts'/name,out/'scripts'/name)
def replace(path,old,new):
    text=path.read_text()
    assert text.count(old)==1,(path,old,text.count(old))
    path.write_text(text.replace(old,new))
p=out/'microduck_deploy/policy.py'
replace(p,'0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,\n    0.0, 0.0873, 0.4579, 0.0049, -0.4530,',
'''0.0, -0.0873, -np.deg2rad(24), -0.0049, np.deg2rad(24), 0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, np.deg2rad(24), 0.0049, -np.deg2rad(24),''')
replace(p,'# Exact HOME_FRAME values, not the ONNX metadata\'s three-decimal rendering.',
'''# Public XgoDuck HOME, consistent with this model's rounded metadata.
# Full user training configuration was not supplied; exact 24-degree values
# follow upstream xgoduck_constants.py. Physical Microduck calibration is retained.''')
replace(p,'/ "hd1910_flat.onnx"','/ "2026-09-24_20-26-30_xgoduck.onnx"')
replace(out/'scripts/supported_home.py',"DEPLOY = Path.home() / 'deploy_hd1910'",'DEPLOY = Path(__file__).resolve().parents[1]')
replace(out/'scripts/home_then_policy.py',"lock = open(ROOT/'logs/supported_sequence.lock', 'a')", "lock = open(Path.home()/'deploy_hd1910/logs/supported_sequence.lock', 'a')")
# Validate this model and config before entering any path that opens hardware.
replace(out/'scripts/home_then_policy.py',"cfg = load_config(ROOT/'robot.json')", "cfg = load_config(ROOT/'robot.json')\n    from verify_profile import verify\n    verify()")
replace(out/'scripts/policy_trial.py','up to 30 seconds','up to 60 seconds')
model=root/'2026-09-24_20-26-30_xgoduck.onnx'
(out/'models').mkdir()
shutil.copy2(model,out/'models'/model.name)
cfg=json.loads((source/'robot.json').read_text())
cfg['model']='models/'+model.name
(out/'robot.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
manifest={'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
 'source_robot_config_sha256':hashlib.sha256((source/'robot.json').read_bytes()).hexdigest(),
 'home_source':'Public XgoDuck HOME: hips/ankles +/-24 degrees; consistent with rounded ONNX metadata, full user training config unavailable',
 'preserved':'Physical joint zeros, directions, IDs, IMU mounting, RAM P5/D0/I0, 50Hz, raw action history, limits; no action filter',
 'stop_behavior':'Completion/error/Ctrl+C turns torque OFF as in existing deployment',
 'deployment_status':'Offline validated; no real motion started'}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(out)
