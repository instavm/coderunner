#!/bin/bash
# Executed inside the guest over SSH. Source is copied, never host-mounted.
set -euo pipefail
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "$HOME/.coderunner/app"
if ! command -v uv >/dev/null; then
    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.9.9/install.sh -o /tmp/coderunner-install-uv.sh
    sh /tmp/coderunner-install-uv.sh
    rm /tmp/coderunner-install-uv.sh
fi
# Use macOS curl for GitHub downloads: some networks reject uv's TLS handshake.
# Pin and verify the standalone interpreter before installing it via uv's local mirror.
python_version=3.13.9
if ! uv python find --managed-python "$python_version" >/dev/null 2>&1; then
    mirror="$HOME/.coderunner/bootstrap/python"
    archive="$mirror/20251031/cpython-3.13.9+20251031-aarch64-apple-darwin-install_only_stripped.tar.gz"
    mkdir -p "$(dirname "$archive")"
    if [ ! -f "$archive" ]; then
        curl --proto '=https' --tlsv1.2 -fLsS --retry 3 --max-time 300 \
            'https://github.com/astral-sh/python-build-standalone/releases/download/20251031/cpython-3.13.9%2B20251031-aarch64-apple-darwin-install_only_stripped.tar.gz' \
            -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    printf '%s  %s\n' '6924bd5fbf74dba25cf733a91deebe313e91f26f261aec8c1d0dc4cb8a9d8738' "$archive" | shasum -a 256 -c -
    UV_PYTHON_INSTALL_MIRROR="file://$mirror" uv python install "$python_version"
    rm "$archive"
fi
uv venv --python "$python_version" --allow-existing "$HOME/.coderunner/venv"
uv pip install --python "$HOME/.coderunner/venv/bin/python" --require-hashes -r macos/requirements.txt
"$HOME/.coderunner/venv/bin/python" -m bash_kernel.install
"$HOME/.coderunner/venv/bin/python" -m playwright install chromium
mkdir -p "$HOME/.coderunner/uploads/skills/user" "$HOME/.coderunner/logs"
cp -R skills/public "$HOME/.coderunner/uploads/skills/"
"$HOME/.coderunner/venv/bin/python" - <<'PY'
import getpass, pathlib, plistlib
home = pathlib.Path.home()
root = home / '.coderunner'
config = {
    'Label': 'io.instavm.coderunner',
    'UserName': getpass.getuser(),
    'ProgramArguments': [str(root / 'venv/bin/python'), str(root / 'app/macos/serve.py')],
    'WorkingDirectory': str(root / 'app'),
    'RunAtLoad': True,
    'KeepAlive': True,
    'StandardOutPath': str(root / 'logs/server.log'),
    'StandardErrorPath': str(root / 'logs/server.log'),
    'EnvironmentVariables': {'HOME': str(home), 'PATH': f'{root}/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'},
}
path = root / 'io.instavm.coderunner.plist'
path.write_bytes(plistlib.dumps(config))
PY
service="system/io.instavm.coderunner"
sudo -n install -o root -g wheel -m 644 "$HOME/.coderunner/io.instavm.coderunner.plist" /Library/LaunchDaemons/io.instavm.coderunner.plist
sudo -n launchctl bootout "$service" 2>/dev/null || true
sudo -n launchctl bootstrap system /Library/LaunchDaemons/io.instavm.coderunner.plist
sudo -n launchctl kickstart "$service"
