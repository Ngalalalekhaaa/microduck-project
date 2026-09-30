
import json,time,dataclasses
from microduck_deploy.servo import ServoBus
config=json.load(open('robot.json'))['serial']
ids=[6,7,8,9,15]
rows=[]
with ServoBus(**config) as bus:
    for cycle in range(100):
        started=time.monotonic()
        for sid in ids:
            row={'cycle':cycle+1,'id':sid}
            t=time.monotonic()
            try:row['feedback']=dataclasses.asdict(bus.read_feedback(sid))
            except Exception as exc:row['error']={'type':type(exc).__name__,'message':str(exc)}
            row['duration_ms']=round((time.monotonic()-t)*1000,3)
            rows.append(row)
        time.sleep(max(0,0.1-(time.monotonic()-started)))
print(json.dumps({'read_only':True,'config':config,'ids':ids,'cycles':100,'cycle_interval_s':0.1,'rows':rows},indent=2))
