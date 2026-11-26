#!/bin/bash
# Installs the sandbox's desktop: Xfce on TigerVNC's Xvnc, and Google Chrome, run by the unprivileged user "box".
# Used by the Dockerfile (DESKTOP=1), and on a sandbox that already exists by the daemon (sandbox/keeper.py) and by
# `coderunner browser computer`.
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
# A home made before the account (a reset puts the browser profile back before the desktop is installed) is root's,
# and Xfce's session then dies at once on ~/.ICEauthority; the account owns its home's top folder.
chown "$USER_NAME:" "$(getent passwd "$USER_NAME" | cut -d: -f6)"
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
# The backdrop: a vertical gradient xfdesktop draws itself (no image file), deep navy at the top to the app's blue
# at the bottom; desktop.py holds the same values and applies them once to a desktop set up before this. An
# owner's own backdrop is kept: only missing settings, or the stock xfce-shapes, are replaced.
desktop_xml="$home/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-desktop.xml"
if [ ! -s "$desktop_xml" ] || grep -q xfce-shapes "$desktop_xml"; then
    {
        echo '<?xml version="1.0" encoding="UTF-8"?>'
        echo '<channel name="xfce4-desktop" version="1.0">'
        echo '  <property name="backdrop" type="empty">'
        echo '    <property name="screen0" type="empty">'
        echo '      <property name="monitorVNC-0" type="empty">'
        for n in 0 1 2 3; do
            echo "        <property name=\"workspace$n\" type=\"empty\">"
            echo '          <property name="image-style" type="int" value="0"/>'
            echo '          <property name="color-style" type="int" value="2"/>'
            for pair in "rgba1 0.059 0.090 0.200 1" "rgba2 0.200 0.380 0.870 1"; do
                read -r name a b c d <<< "$pair"
                echo "          <property name=\"$name\" type=\"array\">"
                for v in "$a" "$b" "$c" "$d"; do echo "            <value type=\"double\" value=\"$v\"/>"; done
                echo '          </property>'
            done
            echo '        </property>'
        done
        echo '      </property>'
        echo '    </property>'
        echo '  </property>'
        echo '</channel>'
    } > "$desktop_xml"
    touch "$home/.config/xfce4/.coderunner-backdrop"
fi
chown -R "$USER_NAME:$USER_NAME" "$home/.config"
chown "$USER_NAME:$USER_NAME" /workspace 2>/dev/null || true
mkdir -p /app/desktop && chmod 700 /app/desktop
echo "Desktop installed: Xfce, TigerVNC, $(/usr/bin/google-chrome-stable --version)"
