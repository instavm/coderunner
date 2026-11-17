#!/usr/bin/env python3
"""Supervise guest-only services; nothing listens on a guest LAN interface."""
import os
import contextlib
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    root = pathlib.Path.home() / '.coderunner'
    uploads = root / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CODERUNNER_SHARED_DIR=str(uploads),
               CODERUNNER_API_TOKEN=(root / 'api-token').read_text().strip(),
               PYTHONUNBUFFERED='1')
    children = []

    def stop(_signum=None, _frame=None):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        children.append(subprocess.Popen([
            sys.executable, '-m', 'jupyter', 'server', '--ip=127.0.0.1', '--port=8888',
            '--ServerApp.port_retries=0', '--no-browser', '--IdentityProvider.token=',
            '--ServerApp.disable_check_xsrf=True', f'--ServerApp.root_dir={uploads}',
        ], cwd=uploads, env=env, start_new_session=True))
        for _ in range(90):
            if children[0].poll() is not None:
                raise RuntimeError('Jupyter exited during startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8888/api/status', timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Jupyter did not become ready within 90 seconds')
        children.append(subprocess.Popen([
            sys.executable, '-m', 'playwright', 'run-server', '--host', '127.0.0.1', '--port', '3000',
        ], env=env, start_new_session=True))
        children.append(subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1',
            '--port', '8222', '--workers', '1', '--no-access-log',
        ], cwd=root / 'app', env=env, start_new_session=True))
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError('A guest service exited; launchd will restart the service group')
    finally:
        for child in children:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
