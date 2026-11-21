#!/bin/bash
# Installs the sandbox's desktop: Xfce on TigerVNC's Xvnc, and Google Chrome, run by the unprivileged user "box".
# Used by the Dockerfile (DESKTOP=1) and by `coderunner browser computer` on a sandbox that already exists.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
USER_NAME="${CODERUNNER_DESKTOP_USER:-box}"

apt-get update
apt-get install -y --no-install-recommends \
    xfce4-session xfwm4 xfce4-panel xfdesktop4 xfce4-settings xfce4-terminal thunar \
    dbus-x11 at-spi2-core tigervnc-standalone-server tigervnc-tools xdotool \
    fonts-liberation fonts-noto-core fonts-noto-color-emoji adwaita-icon-theme ca-certificates curl

if [ ! -x /usr/bin/google-chrome-stable ]; then
    arch="$(dpkg --print-architecture)"            # Google publishes amd64 and arm64 builds
    work="$(mktemp -d)"
    curl -fsSL -o "$work/chrome.deb" "https://dl.google.com/linux/direct/google-chrome-stable_current_${arch}.deb"
    apt-get install -y --no-install-recommends "$work/chrome.deb"
    rm -rf "$work"
fi
apt-get clean
rm -rf /var/lib/apt/lists/*

# Chrome warns about the flag that keeps navigator.webdriver false; the warning bar is turned off by policy.
mkdir -p /etc/opt/chrome/policies/managed
echo '{"CommandLineFlagSecurityWarningsEnabled": false}' > /etc/opt/chrome/policies/managed/coderunner.json

id -u "$USER_NAME" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$USER_NAME"
home="$(getent passwd "$USER_NAME" | cut -d: -f6)"
mkdir -p /workspace "$home/.config/xfce4/xfconf/xfce-perchannel-xml"
# The default panel layout up front: otherwise the first start asks which layout to use.
cp -n /etc/xdg/xfce4/panel/default.xml "$home/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml" || true
# No screen locking or blanking: the owner watches this screen, nobody sits at it.
cat > "$home/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-power-manager.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-power-manager" version="1.0">
  <property name="xfce4-power-manager" type="empty">
    <property name="dpms-enabled" type="bool" value="false"/>
    <property name="blank-on-ac" type="int" value="0"/>
  </property>
</channel>
EOF
chown -R "$USER_NAME:$USER_NAME" "$home/.config"
chown "$USER_NAME:$USER_NAME" /workspace 2>/dev/null || true
mkdir -p /app/desktop && chmod 700 /app/desktop
echo "Desktop installed: Xfce, TigerVNC, $(/usr/bin/google-chrome-stable --version)"
