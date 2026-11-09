#!/usr/bin/env python3
"""Generate online TTS, decode to PCM WAV and peak-normalize a short spoken clip."""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave


def normalize(source, output, peak=.60, padding=.5):
    """Preserve stereo PCM structure; reject silence rather than amplifying it."""
    if not 0 < peak <= .60 or not 0 <= padding <= 5:
        raise ValueError('peak 必须为 0..0.60，padding 必须为 0..5 秒。')
    with wave.open(str(source), 'rb') as wav:
        if (wav.getsampwidth(), wav.getnchannels(), wav.getframerate(), wav.getcomptype()) != (2, 2, 48000, 'NONE'):
            raise ValueError('输入必须为 PCM s16le、48000 Hz、双声道。')
        pcm = array('h', wav.readframes(wav.getnframes()))
    if sys.byteorder != 'little':
        pcm.byteswap()
    maximum = max((abs(x) for x in pcm), default=0)
    if maximum == 0:
        raise ValueError('语音为空或全部静音，未生成输出。')
    gain = (32767 * peak) / maximum
    silence = array('h', [0]) * (round(48000 * padding) * 2)
    normalized = array('h', silence)
    normalized.extend(round(sample * gain) for sample in pcm)
    normalized.extend(silence)
    if sys.byteorder != 'little':
        normalized.byteswap()
    with wave.open(str(output), 'wb') as wav:
        wav.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
        wav.writeframes(normalized.tobytes())
    return {'source_peak_fraction': maximum / 32768, 'target_peak_fraction': peak,
            'seconds': len(normalized) / 2 / 48000, 'padding_seconds_each_end': padding}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default='你好，我是 microduck。')
    parser.add_argument('--voice', default='zh-CN-XiaoxiaoNeural')
    parser.add_argument('--rate', default='-5%')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        if output.exists():
            raise ValueError(f'输出已存在：{output}；请使用新文件名。')
        if not args.text.strip():
            raise ValueError('文本不能为空。')
        for tool in ('edge-tts', 'ffmpeg'):
            if not shutil.which(tool):
                raise ValueError(f'缺少 {tool}；见 README 的可选语音生成依赖。')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='microduck-tts-') as temporary:
            directory = Path(temporary)
            mp3, decoded, generated = (directory / name for name in ('speech.mp3', 'decoded.wav', 'normalized.wav'))
            print('正在通过 Microsoft Edge 在线服务生成语音；输入文本会发送给该服务。', flush=True)
            subprocess.run(['edge-tts', '--voice', args.voice, '--rate=' + args.rate,
                            '--text', args.text, '--write-media', str(mp3)], check=True, timeout=180)
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(mp3), '-ar', '48000',
                            '-ac', '2', '-c:a', 'pcm_s16le', str(decoded)], check=True, timeout=60)
            report = normalize(decoded, generated)
            # Exclusive create prevents an accidental overwrite, including a concurrent writer.
            with output.open('xb') as destination:
                destination.write(generated.read_bytes())
        print(json.dumps(dict(output=str(output), voice=args.voice, **report), ensure_ascii=False, indent=2))
    except (OSError, ValueError, wave.Error, subprocess.SubprocessError) as exc:
        print(f'生成失败：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
