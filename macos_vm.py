#!/usr/bin/env python3
"""Experimental macOS guest backend. Host dependency: Python 3.10+ and Tart 2.41+."""
import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ('ghcr.io/cirruslabs/macos-sequoia-base@sha256:'
                 '8386ef670c04b38e57bd0e97fe2cde6bc0598e1bcfba0212ecba1814e659a6b1')


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(data)
    path.chmod(0o600)


def validate_name(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}', value):
        raise argparse.ArgumentTypeError('Use 1-48 letters, digits, underscores or hyphens')
    return value


class MacOSVM:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.state_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.state = self.root / args.name
        self.state.mkdir(exist_ok=True, mode=0o700)
        self.state.chmod(0o700)
        self.manifest = self.state / 'vm.json'
        self.key = self.state / 'id_ed25519'
        self.token = self.state / 'api-token'
        # macOS allows 104 bytes including OpenSSH's temporary random suffix.
        # Its default $TMPDIR is too long, so use a private directory under /tmp.
        ident = hashlib.sha256(str(self.state).encode()).hexdigest()[:20]
        self.control_dir = Path('/tmp') / f'coderunner-macos-{os.getuid()}-{ident}'
        if self.control_dir.is_symlink():
            raise RuntimeError('Refusing a symlink at the SSH control directory')
        self.control_dir.mkdir(mode=0o700, exist_ok=True)
        if self.control_dir.stat().st_uid != os.getuid():
            raise RuntimeError('SSH control directory has a different owner')
        self.control_dir.chmod(0o700)
        self.control = self.control_dir / 'ssh'
        self.env = dict(os.environ, TART_HOME=str(self.root / 'tart'), TART_NO_AUTO_PRUNE='1')
        self.tart = args.tart

    def tart_run(self, *args, capture=False, **kwargs):
        return run([self.tart, *args], env=self.env, text=True, capture_output=capture, **kwargs)

    def config(self):
        if not self.manifest.exists():
            raise RuntimeError('Create this VM first: python3 macos_vm.py create')
        return json.loads(self.manifest.read_text())

    def running(self):
        rows = json.loads(self.tart_run('list', '--source', 'local', '--format', 'json', capture=True).stdout)
        for row in rows:
            item = {k.lower(): v for k, v in row.items()}
            if item.get('name') == self.args.name:
                return str(item.get('state', '')).lower() == 'running'
        return False

    def ip(self):
        value = self.tart_run('ip', self.args.name, '--wait', '120', capture=True, timeout=130).stdout.strip()
        return str(ipaddress.ip_address(value))

    def ssh_args(self, ip, password=False):
        cfg = self.config()
        return ['ssh', '-F', '/dev/null', '-i', str(self.key),
                '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=5',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                '-o', f'UserKnownHostsFile={self.state / "known_hosts"}',
                '-o', f'HostKeyAlias=coderunner-{self.args.name}',
                '-o', 'StrictHostKeyChecking=accept-new',
                '-o', f'BatchMode={"no" if password else "yes"}',
                f'{cfg["user"]}@{ip}']

    def ssh(self, ip, command, *, data=None, password=False, capture=False, timeout=60):
        env = self.env.copy()
        if password:
            askpass = self.state / 'askpass.sh'
            private_write(askpass, '#!/bin/sh\nprintf "%s\\n" "$CODERUNNER_GUEST_PASSWORD"\n')
            askpass.chmod(0o700)
            env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE='force', DISPLAY=':0',
                       CODERUNNER_GUEST_PASSWORD=os.environ.get('CODERUNNER_GUEST_PASSWORD', 'admin'))
        return run([*self.ssh_args(ip, password), command], env=env,
                   input=data, capture_output=capture, timeout=timeout)

    def create(self):
        if self.manifest.exists():
            raise RuntimeError('VM already configured; use start, stop or status')
        if not 1 <= self.args.cpus <= (os.cpu_count() or 1):
            raise RuntimeError('CPU count must fit the host')
        if self.args.memory < 4096:
            raise RuntimeError('macOS requires at least 4096 MiB RAM')
        host_memory = int(run(['sysctl', '-n', 'hw.memsize'], text=True, capture_output=True).stdout)
        if self.args.memory * 2**20 > host_memory - 4 * 2**30:
            raise RuntimeError('Leave at least 4 GiB RAM for the host')
        if not 1024 <= self.args.port <= 65535:
            raise RuntimeError('Port must be between 1024 and 65535')
        if shutil.disk_usage(self.root).free < 50 * 2**30:
            raise RuntimeError('Allow at least 50 GiB free for a macOS image and guest writes')
        if not 1 <= self.args.download_concurrency <= 32:
            raise RuntimeError('Download concurrency must be between 1 and 32')
        self.tart_run('clone', self.args.image, self.args.name, '--concurrency', str(self.args.download_concurrency))
        self.tart_run('set', self.args.name, '--cpu', str(self.args.cpus), '--memory', str(self.args.memory))
        cfg = {'name': self.args.name, 'image': self.args.image, 'cpus': self.args.cpus,
               'memory_mib': self.args.memory, 'port': self.args.port, 'user': self.args.user,
               'network': self.args.network}
        private_write(self.manifest, json.dumps(cfg, indent=2) + '\n')
        print(f'Created {self.args.name}. Run start to provision CodeRunner.')

    def wait_ssh(self, ip):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((ip, 22), timeout=3):
                    return ip
            except OSError:
                # Changing NAT/host-only networks can leave an old DHCP lease
                # visible briefly while the guest acquires its new address.
                try:
                    value = self.tart_run('ip', self.args.name, capture=True).stdout.strip()
                    ip = str(ipaddress.ip_address(value))
                except (subprocess.CalledProcessError, ValueError):
                    pass
                time.sleep(2)
        raise RuntimeError('Guest SSH did not start within 180 seconds; see vm.log')

    def provision(self, ip):
        if not self.key.exists():
            run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', self.key])
        # Install a per-VM key, then retire the image's public default password.
        test = subprocess.run([*self.ssh_args(ip), 'true'], capture_output=True, timeout=15)
        if test.returncode:
            public_key = self.key.with_suffix('.pub').read_bytes()
            self.ssh(ip, 'umask 077; mkdir -p "$HOME/.ssh"; cat >> "$HOME/.ssh/authorized_keys"',
                     data=public_key, password=True)
        self.ssh(ip, 'true', capture=True)
        self.ssh(ip, 'test "$(uname -s)" = Darwin', capture=True)
        if not (self.state / 'credentials-rotated').exists():
            password_file = self.state / 'guest-password'
            if not password_file.exists():
                private_write(password_file, secrets.token_urlsafe(32) + '\n')
            password = password_file.read_text().strip()
            user = self.config()['user']
            old_password = os.environ.get('CODERUNNER_GUEST_PASSWORD', 'admin')
            # SecureToken/FileVault accounts require the previous password, even
            # for root. Checking the new one also makes a retry safe if SSH was
            # interrupted after the change but before we wrote the marker.
            script = (f'set -e\nif ! dscl . -authonly {user} {shlex.quote(password)} >/dev/null 2>&1; then\n'
                      f'  sudo -n dscl . -passwd /Users/{user} {shlex.quote(old_password)} {shlex.quote(password)}\nfi\n')
            self.ssh(ip, 'bash -s', data=script.encode())
            private_write(self.state / 'credentials-rotated', 'yes\n')
        if not self.token.exists():
            private_write(self.token, secrets.token_urlsafe(32) + '\n')
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode='w:gz') as tar:
                for name in ('server.py', 'auth.py', 'macos', 'skills/public'):
                    tar.add(ROOT / name, arcname=name, filter=lambda entry: None if '__pycache__' in entry.name else entry)
            archive.seek(0)
            run([*self.ssh_args(ip), 'umask 077; mkdir -p "$HOME/.coderunner/app"; tar -xzf - -C "$HOME/.coderunner/app"'], stdin=archive)
        self.ssh(ip, 'umask 077; cat > "$HOME/.coderunner/api-token"', data=self.token.read_bytes())
        self.ssh(ip, 'bash "$HOME/.coderunner/app/macos/guest.sh"', timeout=1800)
        private_write(self.state / 'provisioned', 'yes\n')

    def tunnel(self, ip):
        cfg = self.config()
        if self.control.exists():
            check = subprocess.run(['ssh', '-F', '/dev/null', '-S', str(self.control), '-O', 'check', 'guest'],
                                   capture_output=True)
            if check.returncode == 0:
                return
            self.control.unlink(missing_ok=True)
        args = self.ssh_args(ip)
        run([*args[:-1], '-M', '-S', self.control, '-fNT', '-o', 'ExitOnForwardFailure=yes',
             '-L', f'127.0.0.1:{cfg["port"]}:127.0.0.1:8222', args[-1]])

    def start(self):
        cfg = self.config()
        fresh = not (self.state / 'provisioned').exists()
        network = self.args.network or cfg['network']
        if (fresh or self.args.provision) and network == 'none':
            raise RuntimeError('Provisioning needs internet; use start --network nat, then stop and start --network none')
        if not self.running():
            cmd = [self.tart, 'run', self.args.name, '--no-audio',
                   '--no-clipboard', '--no-usb-accessories']
            if not self.args.gui:
                cmd.append('--no-graphics')
            if network == 'none':
                cmd.append('--net-host')
            with (self.state / 'vm.log').open('a') as log:
                process = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=log, start_new_session=True)
            time.sleep(2)
            if process.poll() is not None:
                raise RuntimeError(f'Tart failed to start; see {self.state / "vm.log"}')
            cfg['network'] = network
            private_write(self.manifest, json.dumps(cfg, indent=2) + '\n')
        elif network != cfg['network']:
            raise RuntimeError('Stop the VM before changing its network mode')
        ip = self.wait_ssh(self.ip())
        if fresh or self.args.provision:
            self.provision(ip)
        self.tunnel(ip)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                if self.request('/health').get('status') == 'ok':
                    print(f'CodeRunner macOS: http://127.0.0.1:{cfg["port"]}/mcp')
                    print(f'Token file: {self.token}')
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError('CodeRunner is not healthy; inspect ~/.coderunner/logs/server.log inside the guest')

    def stop(self):
        if self.control.exists():
            subprocess.run(['ssh', '-F', '/dev/null', '-S', str(self.control), '-O', 'exit', 'guest'],
                           capture_output=True)
        if self.running():
            self.tart_run('stop', self.args.name)
        print('Stopped VM and tunnel; guest files are preserved.')

    def request(self, path, body=None, auth=True):
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer ' + self.token.read_text().strip()
        request = urllib.request.Request(f'http://127.0.0.1:{self.config()["port"]}{path}',
                                         data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)

    def smoke(self):
        try:
            self.request('/execute', {'code': 'print(1)'}, auth=False)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        else:
            raise RuntimeError('Unauthenticated execution was not rejected')
        code = ('import platform, subprocess, pathlib, json; '
                'print(json.dumps({"system": platform.system(), "arch": platform.machine(), '
                '"macos": platform.mac_ver()[0], '
                '"hypervisor": subprocess.check_output(["sysctl", "-n", "kern.hv_vmm_present"], text=True).strip(), '
                '"cwd": str(pathlib.Path.cwd()), "answer": 6 * 7}))')
        result = self.request('/execute', {'code': code})
        if result.get('stderr') or '"system": "Darwin"' not in result.get('stdout', '') or '"hypervisor": "1"' not in result.get('stdout', ''):
            raise RuntimeError(f'Guest execution failed: {result}')
        print(json.dumps({'unauthenticated_status': 401, 'execution': result}, indent=2))

    def status(self):
        print(json.dumps({'config': self.config(), 'running': self.running(),
                          'provisioned': (self.state / 'provisioned').exists(),
                          'state_dir': str(self.state)}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', type=validate_name, default='coderunner-macos')
    p.add_argument('--state-dir', default=os.environ.get('CODERUNNER_MACOS_HOME', '~/.coderunner/macos'))
    p.add_argument('--tart', default=os.environ.get('CODERUNNER_TART', 'tart'))
    sub = p.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create', help='Clone a trusted Tart macOS image (large download)')
    create.add_argument('--image', default=DEFAULT_IMAGE)
    create.add_argument('--download-concurrency', type=int, default=4, help='Parallel image transfers (1-32)')
    create.add_argument('--cpus', type=int, default=2)
    create.add_argument('--memory', type=int, default=4096, help='RAM in MiB')
    create.add_argument('--port', type=int, default=8223)
    create.add_argument('--user', type=validate_name, default='admin')
    create.add_argument('--network', choices=['nat', 'none'], default='nat')
    start = sub.add_parser('start', help='Boot, provision if needed, and open a localhost tunnel')
    start.add_argument('--provision', action='store_true', help='Copy current code and reinstall guest runtime')
    start.add_argument('--gui', action='store_true', help='Show the guest desktop window when booting')
    start.add_argument('--network', choices=['nat', 'none'], help='Persist NAT or host-only networking; requires stopped VM')
    for command in ('stop', 'status', 'smoke'):
        sub.add_parser(command)
    ssh = sub.add_parser('ssh', help='Run a command in the guest using its private key')
    ssh.add_argument('args', nargs=argparse.REMAINDER)
    return p


def main():
    args = parser().parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('macOS guests require an Apple silicon Mac')
    if not shutil.which(args.tart):
        raise RuntimeError('Install Tart first: https://tart.run/quick-start/')
    vm = MacOSVM(args)
    with (vm.state / 'operation.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another operation is already running for this VM') from exc
        if args.command == 'ssh':
            cmd = shlex.join(args.args) if args.args else '$SHELL -l'
            vm.ssh(vm.ip(), cmd, timeout=None)
        else:
            getattr(vm, args.command)()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
