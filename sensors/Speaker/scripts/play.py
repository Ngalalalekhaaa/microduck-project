#!/usr/bin/env python3
"""Play the bundled greeting or a WAV through the HAT speaker with tested levels."""
from __future__ import annotations

import argparse
from array import array
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
CARD = 'aic3104'


def bounded(low, high):
    def parse(value):
        result = int(value)
        if not low <= result <= high:
            raise argparse.ArgumentTypeError(f'范围：{low}..{high}')
        return result
    return parse


def wav_info(path):
    with wave.open(str(path), 'rb') as wav:
        if wav.getcomptype() != 'NONE' or wav.getsampwidth() != 2:
            raise ValueError('请使用 PCM 16-bit WAV。')
        if wav.getnchannels() not in (1, 2) or not wav.getnframes():
            raise ValueError('WAV 必须含有单声道或双声道音频帧。')
        expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
        if len(wav.readframes(wav.getnframes())) != expected:
            raise ValueError('WAV 数据不完整。')
        return wav.getnframes() / wav.getframerate()


def make_melody(path):
    """Public-domain Twinkle melody, 0.40 peak, stereo 48 kHz with gentle fades."""
    rate = 48000
    notes = [('C', 1), ('C', 1), ('G', 1), ('G', 1), ('A', 1), ('A', 1), ('G', 2),
             ('F', 1), ('F', 1), ('E', 1), ('E', 1), ('D', 1), ('D', 1), ('C', 2),
             ('G', 1), ('G', 1), ('F', 1), ('F', 1), ('E', 1), ('E', 1), ('D', 2),
             ('G', 1), ('G', 1), ('F', 1), ('F', 1), ('E', 1), ('E', 1), ('D', 2),
             ('C', 1), ('C', 1), ('G', 1), ('G', 1), ('A', 1), ('A', 1), ('G', 2),
             ('F', 1), ('F', 1), ('E', 1), ('E', 1), ('D', 1), ('D', 1), ('C', 2)]
    frequencies = dict(C=523.251, D=587.330, E=659.255, F=698.456, G=783.991, A=880.0)
    samples = array('h', [0]) * rate  # 0.5 seconds stereo lead-in
    for note, beats in notes:
        count = int(rate * .55 * beats)
        sound_count = max(1, count - int(rate * .035))
        for i in range(sound_count):
            fade = min(1.0, i / (rate * .015), (sound_count - 1 - i) / (rate * .025))
            sample = round(32767 * .40 * fade * math.sin(2 * math.pi * frequencies[note] * i / rate))
            samples.extend((sample, sample))
        samples.extend([0] * ((count - sound_count) * 2))
    samples.extend([0] * rate)
    if sys.byteorder != 'little':
        samples.byteswap()
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
        wav.writeframes(samples.tobytes())


def mixer_commands(pcm):
    return [['amixer', '-q', '-c', CARD, 'cset', 'name=' + name, value]
            for name, value in [('PCM Playback Volume', f'{pcm},{pcm}'),
                                ('Line DAC Playback Volume', '118,118'),
                                ('Line Playback Volume', '0,0'),
                                ('Line Playback Switch', 'on,on')]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav', nargs='?', type=Path, help='默认：你好，我是 microduck。')
    parser.add_argument('--melody', action='store_true', help='生成并播放约 27 秒《小星星》')
    parser.add_argument('--pcm', type=bounded(0, 116), default=116,
                        help='PCM 原始增益值，默认 116；只允许降低到 0..116')
    parser.add_argument('--repeat', type=bounded(1, 10), default=1, help='播放次数，间隔 1 秒')
    parser.add_argument('--dry-run', action='store_true', help='只检查音频并打印命令，不操作声卡')
    args = parser.parse_args(argv)
    if args.wav is not None and args.melody:
        parser.error('自定义 WAV 与 --melody 不能同时使用。')
    try:
        with tempfile.TemporaryDirectory(prefix='microduck-speaker-') as temporary:
            if args.melody:
                path = Path(temporary) / 'twinkle.wav'
                make_melody(path)
            else:
                path = (args.wav or ROOT / 'audio/hello_microduck.wav').expanduser().resolve()
            seconds = wav_info(path)
            print(f'音频：{path}，{seconds:.3f} 秒 × {args.repeat}；PCM={args.pcm}', flush=True)
            commands = mixer_commands(args.pcm)
            playback = ['aplay', '-D', f'plughw:CARD={CARD},DEV=0', str(path)]
            if args.dry_run:
                for command in commands + [playback]:
                    print(shlex.join(command))
                return 0
            for tool in ('amixer', 'aplay'):
                if not shutil.which(tool):
                    raise ValueError(f'缺少 {tool}，请先安装 alsa-utils。')
            for command in commands:
                subprocess.run(command, check=True, timeout=10)
            for index in range(args.repeat):
                if index:
                    time.sleep(1)
                subprocess.run(playback, check=True, timeout=seconds + 15)
    except (OSError, ValueError, wave.Error, subprocess.SubprocessError) as exc:
        print(f'播放失败：{exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
