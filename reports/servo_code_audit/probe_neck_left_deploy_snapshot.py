"""Read-only sequential inspection of IDs 1..9, with ID7 as reference."""
import dataclasses
import json
import time

from microduck_deploy.servo import ServoBus


config = json.load(open("robot.json"))["serial"]
rows = []
with ServoBus(**config) as bus:
    for sid in [7, 1, 2, 3, 4, 5, 6, 8, 9]:
        row = {"id": sid, "timeout_s": bus.timeout, "trials": []}
        for trial in range(3):
            result = {"trial": trial + 1}
            for name, fn in [("ping", bus.ping), ("feedback", bus.read_feedback)]:
                try:
                    value = fn(sid)
                    result[name] = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
                except Exception as exc:
                    result[name] = {"error": type(exc).__name__, "message": str(exc)}
                time.sleep(0.05)
            row["trials"].append(result)
        if any(r["ping"] is True for r in row["trials"]):
            try:
                row["configuration"] = bus.read_configuration(sid)
            except Exception as exc:
                row["configuration"] = {"error": type(exc).__name__, "message": str(exc)}
        rows.append(row)
    print(json.dumps({"read_only": True, "port": config["port"], "servos": rows}, ensure_ascii=False, indent=2))
