
import dataclasses,json,time
from microduck_deploy.servo import ServoBus,ServoTimeout
found=[]
errors=[]
with ServoBus('/dev/ttyS2',1000000,timeout=0.03,register_profile='hls_2') as bus:
    for sid in range(254):
        try:
            bus.ping(sid)
            found.append(sid)
        except ServoTimeout:pass
        except Exception as exc:errors.append({'id':sid,'error':type(exc).__name__,'message':str(exc)})
        time.sleep(0.01)
    readings=[]
    for sid in found:
        row={'id':sid,'trials':[]}
        for n in range(5):
            trial={}
            for name,fn in [('identity',bus.read_identity),('feedback',bus.read_feedback)]:
                try:
                    value=fn(sid)
                    trial[name]=dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
                except Exception as exc:trial[name]={'error':type(exc).__name__,'message':str(exc)}
                time.sleep(0.05)
            row['trials'].append(trial)
        try:row['configuration']=bus.read_configuration(sid)
        except Exception as exc:row['configuration']={'error':type(exc).__name__,'message':str(exc)}
        readings.append(row)
print(json.dumps({'read_only':True,'port':'/dev/ttyS2','baudrate':1000000,'scan_ids':[0,253],'found_ids':found,'non_timeout_errors':errors,'readings':readings},ensure_ascii=False,indent=2))
