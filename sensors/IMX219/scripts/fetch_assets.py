#!/usr/bin/env python3
"""Fetch pinned vendor assets, or --check existing files without network access."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(asset, path, attempts=3):
    """Install only a complete, pinned response; failed attempts leave no asset."""
    headers = {}
    if urllib.parse.urlsplit(asset['url']).hostname == 'cdn.arducam.com':
        # This CDN returns 403 for Python's default UA. Other vendors retain
        # urllib's default request, which has also been checked independently.
        headers['User-Agent'] = 'Mozilla/5.0'
    for attempt in range(attempts):
        temp = None
        try:
            request_headers = headers.copy()
            if attempt:
                # Ask intermediate caches to revalidate; never relax the hash
                # or append alternate URLs that might fetch a different asset.
                request_headers.update({'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
            request = urllib.request.Request(asset['url'], headers=request_headers)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temp = Path(output.name)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        raise RuntimeError(f'Expected complete HTTP 200 response, received {response.status}')
                    length = response.headers.get('Content-Length')
                    received = 0
                    while chunk := response.read(1024*1024):
                        received += len(chunk)
                        output.write(chunk)
                    if length is not None and received != int(length):
                        raise RuntimeError(f'Incomplete response: {received} bytes; expected {length}')
            actual = digest(temp)
            if actual != asset['sha256']:
                raise RuntimeError(f'SHA256 mismatch: expected {asset["sha256"]}, received {actual}')
            temp.chmod(0o644)
            temp.replace(path)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, RuntimeError) as exc:
            # Missing/forbidden URLs need an explicit fix, not repeated requests.
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (408, 429, 500, 502, 503, 504)
            if not retryable or attempt + 1 == attempts:
                raise SystemExit(f'Download failed: {asset["url"]}: {exc}; no asset installed') from exc
            print(f'RETRY {asset["path"]} ({attempt+1}/{attempts}): {exc}', file=sys.stderr, flush=True)
            time.sleep(attempt + 1)
        finally:
            if temp and temp.exists():
                temp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify only; do not download')
    args = parser.parse_args()
    assets = json.loads((ROOT/'references/sources.json').read_text())['assets']
    for asset in assets:
        path = ROOT/asset['path']
        if not path.resolve().is_relative_to(ROOT):
            raise SystemExit('Invalid asset path')
        if path.exists():
            if digest(path) != asset['sha256']:
                raise SystemExit(f'SHA256 mismatch: {path}; preserve it for inspection, do not install')
            print('OK', asset['path'])
            continue
        if args.check:
            raise SystemExit(f'Missing {path}; run without --check to download')
        path.parent.mkdir(parents=True, exist_ok=True)
        download(asset, path)
        print('DOWNLOADED', asset['path'], flush=True)


if __name__ == '__main__':
    main()
