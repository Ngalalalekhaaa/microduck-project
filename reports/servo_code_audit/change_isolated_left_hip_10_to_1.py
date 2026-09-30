"""One-use, user-authorized ID change for the isolated left hip. No motion writes."""
import dataclasses
import json
import time
from pathlib import Path

from microduck_deploy.servo import ServoBus, ServoTimeout, build_packet


OLD_ID, NEW_ID = 10, 1

log = {"operation": "isolated_left_hip_id_10_to_1", "events": [], "success": False}
destination = Path("logs/left_hip_id_10_to_1_20260929.json")


def save():
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(log, ensure_ascii=False, indent=2))


def scan(bus):
    found = []
    for sid in range(254):
        try:
            bus.ping(sid)
            found.append(sid)
        except ServoTimeout:
            pass
        time.sleep(0.005)
    return found


def write_allowed(bus, sid, address, value, acknowledge=True):
    if (sid, address, value) not in {(OLD_ID, 55, 0), (OLD_ID, 5, NEW_ID), (OLD_ID, 55, 1), (NEW_ID, 55, 1)}:
        raise RuntimeError("Write outside the fixed ID-change sequence")
    packet = build_packet(sid, 3, bytes([address, value]))
    log["events"].append({"write_hex": packet.hex(), "id": sid, "address": address, "value": value})
    save()
    with bus._io:
        deadline = bus._send(packet)
        if acknowledge:
            bus._read_status(deadline, sid, 0)
    time.sleep(0.1)


try:
    with ServoBus("/dev/ttyS2", 1000000, timeout=0.05, register_profile="hls_2") as bus:
        log["before_scan"] = scan(bus)
        if log["before_scan"] != [OLD_ID]:
            raise RuntimeError("Expected only ID10 online; refusing ID change")
        before = bus.read(OLD_ID, 0, 40)
        log["before_registers_0_39_hex"] = before.hex()
        log["before_configuration"] = bus.read_configuration(OLD_ID)
        log["before_feedback"] = dataclasses.asdict(bus.read_feedback(OLD_ID))
        if log["before_configuration"]["torque"] != 0 or log["before_configuration"]["lock"] != 1:
            raise RuntimeError("Expected torque off and EEPROM locked")
        if before[3:5] != bytes([10, 31]) or before[5] != OLD_ID:
            raise RuntimeError("Identity changed since inspection")
        save()  # Persist configuration snapshot before the first write.
        try:
            write_allowed(bus, OLD_ID, 55, 0)
            if bus.read(OLD_ID, 55, 1) != b"\x00":
                raise RuntimeError("Unlock verification failed")
            # ID-change acknowledgement can carry the old or new ID.
            # Do not infer success from it; verify the new identity directly.
            write_allowed(bus, OLD_ID, 5, NEW_ID, acknowledge=False)
            log["new_identity"] = bus.read_identity(NEW_ID)
        finally:
            relocked = []
            for sid in (NEW_ID, OLD_ID):
                try:
                    bus.read_identity(sid)
                except ServoTimeout:
                    continue
                write_allowed(bus, sid, 55, 1)
                if bus.read(sid, 55, 1) != b"\x01":
                    raise RuntimeError("Relock verification failed")
                relocked.append(sid)
            log["relocked_ids"] = relocked
            save()
            if not relocked:
                raise RuntimeError("Unable to find/relock servo after ID operation")
        after = bus.read(NEW_ID, 0, 40)
        log["after_registers_0_39_hex"] = after.hex()
        changed = [i for i in range(40) if before[i] != after[i]]
        log["changed_registers_0_39"] = changed
        if after[5] != NEW_ID or any(i not in (5, 7) for i in changed):
            raise RuntimeError("Unexpected EEPROM readback differences")
        log["after_configuration"] = bus.read_configuration(NEW_ID)
        if log["after_configuration"]["torque"] != 0:
            raise RuntimeError("Unexpected torque state")
        log["verification"] = []
        for _ in range(5):
            log["verification"].append({"identity": bus.read_identity(NEW_ID),
                                        "feedback": dataclasses.asdict(bus.read_feedback(NEW_ID))})
            time.sleep(0.05)
        log["after_scan"] = scan(bus)
        if log["after_scan"] != [NEW_ID]:
            raise RuntimeError("Expected only ID1 after modification")
        log["success"] = True
except Exception as exc:
    log["error"] = {"type": type(exc).__name__, "message": str(exc)}
finally:
    save()
    print(json.dumps(log, ensure_ascii=False, indent=2))
if not log["success"]:
    raise SystemExit(1)
