#!/usr/bin/env bash
# 在电脑上执行：bash scripts/capture_frame.sh [--bright] USER@IP [保存目录]
# 默认有限轮自动曝光；--bright 使用稍亮的目标，不再固定极高曝光和增益。
# rkaiq 负责白平衡/颜色，须使用仓库安装脚本的 AE-disabled 配置。
# 摄像头先开流再设置曝光；结束后停流并恢复原控制值。不操作舵机。
set -euo pipefail

bright=0
if [[ ${1:-} == --bright ]]; then bright=1; shift; fi
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    echo "用法：bash $0 [--bright] USER@IP [保存目录]"
    echo "默认自动曝光；--bright 使用稍亮的自动曝光目标。需在电脑安装 ssh、scp、python3、ffmpeg。"
    exit 0
fi
if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[a-zA-Z_][a-zA-Z0-9_.-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]]; then
    echo "用法：bash $0 [--bright] USER@IP [保存目录]（主机名或 IPv4）" >&2
    exit 2
fi
host=$1
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${2:-"$root/captures/$(date +%Y%m%d)/frame_$(date +%H%M%S)_$$"}
for tool in ssh scp python3 ffmpeg; do
    command -v "$tool" >/dev/null || { echo "电脑缺少命令：$tool" >&2; exit 1; }
done
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
for artifact in frame.png frame.nv12 capture.json capture.log ssh.log transfer.log; do
    if [[ -e "$out/$artifact" || -L "$out/$artifact" ]]; then
        echo "保存目录已有拍摄产物：$out/$artifact；请换一个空目录，避免覆盖或误认旧照片。" >&2
        exit 2
    fi
done
opts=(-o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

echo "连接 $host，自动曝光并抓取一帧；SSH 如提示密码，请输入 Radxa 登录密码。"
# 代码通过 SSH 标准输入发送；密码由 SSH 从终端读取，不写入脚本或日志。
if ssh -T "${opts[@]}" "$host" python3 - "$bright" >"$out/capture.json" 2> >(tee "$out/ssh.log" >&2) <<'PY'
from pathlib import Path
import hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time, traceback
from datetime import datetime, timezone, timedelta

out = Path(tempfile.mkdtemp(prefix='microduck-camera-'))
log = (out / 'capture.log').open('w', buffering=1)
bright = sys.argv[1] == '1'
result = dict(success=False, remote_log=str(out/'capture.log'),
              bright=bright, ae_history=[], warnings=[],
              controls_readback_kind='V4L2 cache; not sensor register verification')
original, limits = {}, {}
proc = None
changed = False
sequence = out / 'sequence.nv12'
width, height = 640, 480
size = width * height * 3 // 2
max_frames = 240  # At most 110,592,000 temporary bytes; deleted after capture.

def run(args, required=True):
    print('COMMAND', args, file=log)
    r = subprocess.run(args, capture_output=True, text=True, timeout=20)
    print(r.stdout, r.stderr, 'EXIT', r.returncode, file=log)
    if required and r.returncode:
        raise RuntimeError(f'{args[0]} 失败：{r.stdout}\n{r.stderr}')
    return r.stdout + r.stderr, r.returncode

def force_controls(values):
    # Vendor stream-on rewrites exposure. Equal cached values may suppress writes:
    # force a legal adjacent value first, then the intended value, after stream-on.
    near = {k: v-1 if v > limits[k][0] else v+1 for k, v in values.items()}
    for controls in (near, values):
        run(['v4l2-ctl', '-d', device, '--set-ctrl=' +
             ','.join(k+'='+str(v) for k, v in controls.items())])

def frames():
    return sequence.stat().st_size // size if sequence.exists() else 0

def wait_frames(target, seconds=0):
    if target > max_frames:
        raise RuntimeError('已达到有限帧数上限，未能完成曝光收敛；详见 ae_history')
    start = time.monotonic()
    while frames() < target or time.monotonic()-start < seconds:
        if proc.poll() is not None:
            raise RuntimeError('取流提前退出（可能摄像头占用、权限或 I/O 错误），详见 capture.log')
        if time.monotonic()-start > 15:
            raise TimeoutError('等待摄像头新帧超时')
        time.sleep(0.025)

def latest_frame():
    n = frames()
    if n < 1:
        raise RuntimeError('摄像头没有输出完整帧')
    with sequence.open('rb') as stream:
        stream.seek((n-1)*size)
        data = stream.read(size)
    if len(data) != size:
        raise RuntimeError('NV12 帧长度不正确')
    return n, data

def measure(data):
    y = data[:width*height]
    # Sample the centre and the whole frame; retain whole-frame clipping figures.
    centre = b''.join(y[row*width+width//5:row*width+width*4//5]
                      for row in range(height//5, height*4//5, 4))
    y_mean = sum(y)/len(y)
    centre_mean = sum(centre)/len(centre)
    return dict(y_min=min(y), y_max=max(y), y_mean=y_mean,
                centre_y_mean=centre_mean, metered_y=0.7*centre_mean+0.3*y_mean,
                clipped_high_ratio=sum(v >= white-2 for v in y)/len(y),
                clipped_low_ratio=sum(v <= black+2 for v in y)/len(y))

def record(stage, controls):
    n, data = latest_frame()
    m = measure(data)
    entry = dict(stage=stage, frame=n, seconds=round(time.monotonic()-started, 3),
                 requested_controls=controls.copy(), **m)
    result['ae_history'].append(entry)
    print('AE', json.dumps(entry), file=log)
    return m, data

def on_signal(signum, frame):
    raise RuntimeError(f'抓帧被信号 {signum} 中止')

signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
try:
    if not shutil.which('v4l2-ctl'):
        raise RuntimeError('Radxa 缺少 v4l2-ctl，请先安装 v4l-utils')
    nodes = [p.parent.name for p in Path('/sys/class/video4linux').glob('video*/name')
             if p.read_text().strip() == 'rkisp_mainpath']
    if len(nodes) != 1:
        raise RuntimeError(f'找不到唯一 rkisp_mainpath：{nodes}')
    device = '/dev/' + nodes[0]
    result['device'] = device
    if not os.access(device, os.R_OK | os.W_OK):
        raise RuntimeError(f'无权读写 {device}；请把登录用户加入 video 组后重新登录')
    if shutil.which('fuser'):
        capture_nodes = ['/dev/'+p.parent.name
                         for p in Path('/sys/class/video4linux').glob('video*/name')
                         if p.read_text().strip() in ('rkisp_mainpath', 'rkisp_selfpath')]
        busy, rc = run(['fuser', *capture_nodes], required=False)
        if rc == 0:
            raise RuntimeError(f'摄像头正被其他程序占用，请先结束其拍摄：{busy.strip()}')
    _, rc = run(['systemctl', 'is-active', '--quiet', 'rkaiq_3A.service'], required=False)
    result['rkaiq_active'] = rc == 0
    if rc != 0:
        result['warnings'].append('rkaiq_3A 未运行；可抓帧，但白平衡/颜色可能不正常')
    iq_path = Path('/etc/iqfiles/imx219_rpi-camera-v2_default.json')
    if iq_path.exists():
        iq = iq_path.read_bytes()
        enabled = re.search(rb'"ae_calib"\s*:\s*\{\s*"CommCtrl"\s*:\s*\{\s*"Enable"\s*:\s*([01])', iq)
        result['iq_file'] = str(iq_path)
        result['iq_sha256'] = hashlib.sha256(iq).hexdigest()
        result['rkaiq_ae_enabled_in_file'] = bool(int(enabled[1])) if enabled else None
        if result['rkaiq_active'] and result['rkaiq_ae_enabled_in_file'] is not False:
            raise RuntimeError('rkaiq 的 IQ 配置未确认关闭 AE；请先运行 scripts/setup_isp.py --apply，避免两套曝光控制同时写入')
    elif result['rkaiq_active']:
        raise RuntimeError('rkaiq 正在运行，但缺少预期 IMX219 IQ 文件；无法确认曝光控制归属')
    _, rc = run(['systemctl', 'is-active', '--quiet', 'mediad.service'], required=False)
    if rc == 0:
        raise RuntimeError('mediad 正在运行，可能占用摄像头或同时调整曝光；请先结束该拍摄服务')

    # Read the existing sensor mode only. Do not change media links or restart ISP.
    sensors = [p for p in Path('/sys/class/video4linux').glob('v4l-subdev*/name')
               if 'imx219' in p.read_text().lower()]
    if len(sensors) != 1:
        raise RuntimeError(f'找不到唯一 IMX219 子设备：{sensors}')
    sensor = sensors[0]
    sensor_device = '/dev/' + sensor.parent.name
    sensor_fmt, _ = run(['v4l2-ctl', '-d', sensor_device, '--get-subdev-fmt=pad=0'], required=False)
    match = re.search(r'Width/Height\s*:\s*(\d+)/(\d+)', sensor_fmt)
    if match:
        mode = (int(match[1]), int(match[2]))
    else:
        # Older vendor kernels reject newer SUBDEV_CLIENT_CAP but media-ctl works.
        mode = None
        for media in sorted(Path('/dev').glob('media[0-9]*')):
            topology, _ = run(['media-ctl', '-d', str(media), '-p'], required=False)
            name = re.escape(sensor.read_text().strip())
            block = re.search(r'entity \d+: '+name+r'\b(?:(?!\n- entity).)*', topology, re.S)
            fmt_match = re.search(r'fmt:[^/\s]+/(\d+)x(\d+)', block[0]) if block else None
            if fmt_match:
                mode = (int(fmt_match[1]), int(fmt_match[2]))
                break
    # Verified vendor IMX219 mode timing, not the scaled mainpath output size.
    known_modes = {(1920, 1080): (1766, 1760), (3280, 2464): (2500, 2496)}
    if mode not in known_modes:
        raise RuntimeError(f'未识别的 IMX219 当前模式 {mode}，无法安全确定曝光上限')
    frame_length, max_exposure = known_modes[mode]
    height = 360 if mode == (1920, 1080) else 480
    size = width * height * 3 // 2
    result['sensor_mode'] = dict(device=sensor_device, width=mode[0], height=mode[1],
                                 frame_length_lines=frame_length,
                                 timing_source='vendor IMX219 fixed mode table',
                                 max_exposure_lines=max_exposure)
    text, _ = run(['v4l2-ctl', '-d', device, '--list-ctrls'])
    for name in ('exposure', 'analogue_gain'):
        row = re.search(r'^\s*'+name+r'\s+.*min=(\d+) max=(\d+).*value=(\d+)', text, re.M)
        if not row:
            raise RuntimeError(f'无法读取 {name} 范围与原值')
        limits[name] = (int(row[1]), int(row[2]))
        original[name] = int(row[3])
    result['original_cached_controls'] = original.copy()
    max_exposure = min(max_exposure, limits['exposure'][1])
    min_exposure = max(10, limits['exposure'][0])
    min_gain = max(256, limits['analogue_gain'][0])
    max_gain = min(2048, limits['analogue_gain'][1])  # Limit noise; do not force 11x.
    result['ae_limits'] = dict(exposure=[min_exposure, max_exposure],
                               analogue_gain=[min_gain, max_gain])
    run(['v4l2-ctl', '-d', device,
         f'--set-fmt-video=width={width},height={height},pixelformat=NV12'])
    fmt, _ = run(['v4l2-ctl', '-d', device, '--get-fmt-video'])
    stride = re.search(r'Bytes per Line\s*:\s*(\d+)', fmt)
    if (not stride or int(stride[1]) != width or "'NV12'" not in fmt
            or not re.search(r'Width/Height\s*:\s*'+f'{width}/{height}'+r'\b', fmt)):
        raise RuntimeError('抓帧格式不正确，详见 capture.log')
    full_range = 'Full Range' in fmt
    black, white = (0, 255) if full_range else (16, 235)
    target_y = black + (white-black)*(0.48 if bright else 0.38)
    result.update(color_range='pc' if full_range else 'tv', target_y=target_y,
                  ae_owner='this script; rkaiq AE must be disabled',
                  max_temporary_bytes=size*max_frames)
    args = ['v4l2-ctl', '-d', device, '--stream-mmap=4', '--stream-count='+str(max_frames),
            '--stream-to='+str(sequence), '--verbose']
    print('COMMAND', args, file=log)
    log.flush()
    started = time.monotonic()
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    wait_frames(5)
    desired = dict(exposure=max(min_exposure, min(500, max_exposure)), analogue_gain=min_gain)
    changed = True  # Restore even if the first forced write partly fails.
    force_controls(desired)
    # Continuous streaming lets the ISP AWB settle; each AE iteration uses new frames.
    wait_frames(frames()+12, seconds=2.0)
    status = 'iteration_limit'
    for iteration in range(7):
        m, data = record('ae_'+str(iteration), desired)
        ratio = (target_y-black)/max(1, m['metered_y']-black)
        if abs(m['metered_y']-target_y) <= 10 and m['clipped_high_ratio'] <= 0.05:
            status = 'converged'
            break
        factor = max(0.4, min(2.5, ratio**1.5))
        # Reduce widespread highlight clipping; isolated lamps need not darken all of the image.
        if m['clipped_high_ratio'] > 0.05:
            factor = min(factor, 0.8 if m['clipped_high_ratio'] > 0.1 else 0.95)
        product = desired['exposure'] * desired['analogue_gain']/256 * factor
        exposure = max(min_exposure, min(max_exposure, round(product*256/min_gain)))
        gain = max(min_gain, min(max_gain, round(product*256/exposure)))
        next_controls = dict(exposure=exposure, analogue_gain=gain)
        if next_controls == desired:
            status = 'control_limit'
            break
        if iteration == 6:
            break
        desired = next_controls
        force_controls(desired)
        wait_frames(frames()+12)
    # Give AWB more fresh frames at final exposure; no offline recoloring is performed.
    wait_frames(frames()+20, seconds=0.5)
    m, data = record('final', desired)
    converged = abs(m['metered_y']-target_y) <= 10 and m['clipped_high_ratio'] <= 0.05
    if not converged:
        result['warnings'].append('有限轮自动曝光未达到目标；请结合照片和 ae_history 查看场景明暗')
    raw = out/'frame.nv12'
    raw.write_bytes(data)
    cached, _ = run(['v4l2-ctl', '-d', device, '--get-ctrl=exposure,analogue_gain'])
    result.update(success=True, time_china=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  width=width, height=height, format='NV12', raw_path=str(raw),
                  sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                  requested_controls=desired, final_cached_controls=cached.strip(),
                  controls_applied_after_stream_on=True, ae_converged=converged,
                  ae_stop_reason=status, **m)
except BaseException as exc:
    result.update(success=False, error=str(exc))
    traceback.print_exc(file=log)
finally:
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if changed:
        try:
            force_controls(original)
            restored, _ = run(['v4l2-ctl', '-d', device, '--get-ctrl=exposure,analogue_gain'])
            for name, value in original.items():
                if not re.search(r'^'+name+': '+str(value)+r'$', restored, re.M):
                    raise RuntimeError('曝光参数恢复的缓存读回不一致')
            result['restored_cached_controls'] = original
        except Exception as exc:
            result.update(success=False, restore_error=str(exc))
            traceback.print_exc(file=log)
    sequence.unlink(missing_ok=True)
    (out/'capture.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    log.close()
    print(json.dumps(result, ensure_ascii=False))
    if not result['success']:
        print('抓帧失败：'+result.get('error', result.get('restore_error', '未知错误')), file=sys.stderr)
        print('远端诊断日志：'+str(out/'capture.log'), file=sys.stderr)
    sys.exit(0 if result['success'] else 1)
PY
then
    capture_status=0
else
    capture_status=$?
fi

# Remote errors also return JSON. Retrieve their full diagnostic log before exiting.
if ! remote_log=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_log"])' "$out/capture.json"); then
    echo "连接或远端程序未返回有效结果，请查看：$out/ssh.log" >&2
    exit 1
fi
if ! scp "${opts[@]}" "$host:$remote_log" "$out/capture.log" 2> >(tee -a "$out/transfer.log" >&2); then
    echo "未能下载日志，远端仍保留：$remote_log；本地状态：$out/capture.json" >&2
    exit 1
fi
if [[ $capture_status -ne 0 ]]; then
    echo "拍照未成功；完整日志：$out/capture.log；状态：$out/capture.json" >&2
    exit "$capture_status"
fi
remote_raw=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["raw_path"])' "$out/capture.json")
scp "${opts[@]}" "$host:$remote_raw" "$out/frame.nv12" 2> >(tee -a "$out/transfer.log" >&2)
python3 - "$out" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads((p/'capture.json').read_text())
data = (p/'frame.nv12').read_bytes()
if len(data) != r['bytes'] or hashlib.sha256(data).hexdigest() != r['sha256']:
    raise SystemExit('传回文件校验失败')
print(f"帧亮度 Y：最小 {r['y_min']}，最大 {r['y_max']}，平均 {r['y_mean']:.2f}")
print(f"自动曝光收敛：{r['ae_converged']}；参数：{r['requested_controls']}")
for warning in r['warnings']:
    print('提示：'+warning)
PY
color_range=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["color_range"])' "$out/capture.json")
video_size=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(str(r["width"])+"x"+str(r["height"]))' "$out/capture.json")
# Only decode NV12 to PNG; do not artificially adjust color or brightness after capture.
ffmpeg -nostdin -hide_banner -loglevel error -y -f rawvideo -pixel_format nv12 \
    -video_size "$video_size" -color_range "$color_range" -i "$out/frame.nv12" -frames:v 1 "$out/frame.png"
echo "图片：$out/frame.png"
echo "原始 NV12 帧：$out/frame.nv12"
echo "日志：$out/capture.log"
echo "曝光历程与参数：$out/capture.json"
