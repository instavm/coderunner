"""Consistent desktop Chrome identity for the sandbox and headless browsers.

The sandbox VM runs Google Chrome on Linux arm64/x86_64 with a SwiftShader
software GPU and often a POSIX locale. Anti-bot systems (Cloudflare, Akamai,
DataDome, PerimeterX, AWS WAF) score that combination — Linux aarch64 User-Agent,
mismatched Client Hints, SwiftShader WebGL renderer, headless flags, or UTC
timezone — as automated traffic.

This module derives a self-consistent macOS desktop Google Chrome profile from
the *actual* Chrome version running inside the browser so the TLS/HTTP2 stack,
User-Agent header, Sec-CH-UA Client Hints, navigator.userAgentData, platform,
locale, timezone, and WebGL renderer all agree.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any

DEFAULT_CHROME_FULL = "136.0.0.0"
DEFAULT_CHROME_MAJOR = "136"
MAC_PLATFORM_VERSION = "15.4.0"
DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
DEFAULT_LOCALE = "en-US"

# Chromium's own GREASE brand rotation for Sec-CH-UA (embedder_support::GetUserAgentBrandList).
_GREASE_BRANDS = (
    ("Not(A:Brand", "99"),
    ("Not/A)Brand", "8"),
    ("Not;A=Brand", "24"),
)

_VERSION_RE = re.compile(r"(?:Chrome|HeadlessChrome|Chromium)/(\d+)(?:\.(\d+)\.(\d+)\.(\d+))?")
_BARE_VERSION_RE = re.compile(r"^(\d{2,4})(?:\.(\d+)\.(\d+)\.(\d+))?$")


def parse_chrome_version(product_or_ua: str | None) -> tuple[str, str]:
    """Return (full_version, major_version) from a CDP Browser.getVersion product or UA string."""
    text = str(product_or_ua or "").strip()
    m = _VERSION_RE.search(text) or _BARE_VERSION_RE.match(text)
    if not m:
        return DEFAULT_CHROME_FULL, DEFAULT_CHROME_MAJOR
    major = m.group(1)
    if m.group(2) is not None:
        full = f"{major}.{m.group(2)}.{m.group(3)}.{m.group(4)}"
    else:
        full = f"{major}.0.0.0"
    return full, major


def desktop_user_agent(product_or_ua: str | None = None) -> str:
    """Standard reduced desktop macOS Chrome User-Agent matching the actual Chrome major version."""
    _, major = parse_chrome_version(product_or_ua)
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    )


def _grease_brand(major: str) -> tuple[str, str]:
    try:
        idx = int(major) % len(_GREASE_BRANDS)
    except ValueError:
        idx = 0
    return _GREASE_BRANDS[idx]


def chrome_identity(product_or_ua: str | None = None, *, accept_language: str | None = None) -> dict[str, Any]:
    """Build CDP Emulation.setUserAgentOverride parameters for a regular desktop Chrome."""
    full, major = parse_chrome_version(product_or_ua)
    grease_name, grease_ver = _grease_brand(major)
    brands = [
        {"brand": "Chromium", "version": major},
        {"brand": "Google Chrome", "version": major},
        {"brand": grease_name, "version": grease_ver},
    ]
    full_list = [
        {"brand": "Chromium", "version": full},
        {"brand": "Google Chrome", "version": full},
        {"brand": grease_name, "version": f"{grease_ver}.0.0.0"},
    ]
    # CDP Emulation.setUserAgentOverride expects bare comma-separated language codes ("en-US,en")
    # because Chromium automatically formats ;q= weights on the Accept-Language header.
    cdp_lang = (accept_language or host_accept_language()).split(";", 1)[0]
    if "," not in cdp_lang:
        cdp_lang = "en-US,en"
    return {
        "userAgent": desktop_user_agent(full),
        "acceptLanguage": cdp_lang,
        "platform": "MacIntel",
        "userAgentMetadata": {
            "brands": brands,
            "fullVersionList": full_list,
            "fullVersion": full,
            "platform": "macOS",
            "platformVersion": MAC_PLATFORM_VERSION,
            "architecture": "arm",
            "model": "",
            "mobile": False,
            "bitness": "64",
            "wow64": False,
        },
    }


def chrome_headers(product_or_ua: str | None = None, *, accept_language: str | None = None) -> dict[str, str]:
    """Realistic top-level Chrome navigation HTTP headers for lightweight HTTP fetches (web_read)."""
    ident = chrome_identity(product_or_ua, accept_language=accept_language)
    brands = ident["userAgentMetadata"]["brands"]
    sec_ch_ua = ", ".join(f'"{b["brand"]}";v="{b["version"]}"' for b in brands)
    return {
        "User-Agent": ident["userAgent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": accept_language or host_accept_language(),
        "Sec-CH-UA": sec_ch_ua,
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"macOS"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def host_timezone() -> str | None:
    """Detect the host Mac/Linux IANA timezone (e.g. 'America/Los_Angeles') so the VM matches the owner's IP."""
    tz = (os.environ.get("TZ") or "").strip().lstrip(":")
    if tz and "/" in tz and not tz.startswith("/"):
        return tz
    localtime = pathlib.Path("/etc/localtime")
    try:
        resolved = str(localtime.resolve())
        marker = "zoneinfo/"
        if marker in resolved:
            zone = resolved.split(marker, 1)[1].strip()
            if zone and "/" in zone:
                return zone
    except OSError:
        pass
    etc_tz = pathlib.Path("/etc/timezone")
    try:
        if etc_tz.is_file():
            zone = etc_tz.read_text(encoding="utf-8").strip()
            if zone and "/" in zone:
                return zone
    except OSError:
        pass
    return None


def host_accept_language() -> str:
    """Return a clean desktop Accept-Language header matching the host locale when standard, else en-US."""
    raw = (os.environ.get("LANG") or os.environ.get("LC_ALL") or "").split(".", 1)[0].strip()
    if raw and raw.upper() not in ("C", "POSIX") and "_" in raw:
        lang, region = raw.split("_", 1)
        if len(lang) == 2 and len(region) == 2 and lang.isalpha() and region.isalpha():
            tag = f"{lang.lower()}-{region.upper()}"
            if tag == "en-US":
                return DEFAULT_ACCEPT_LANGUAGE
            return f"{tag},{lang.lower()};q=0.9,en-US;q=0.8,en;q=0.7"
    return DEFAULT_ACCEPT_LANGUAGE


# Injected ONLY into the sandbox VM Chrome or headless Chrome before any page script executes.
# Fixes software-VM / headless telltales (SwiftShader WebGL, missing window.chrome.runtime,
# empty plugins, low vCPU count, zero outerWidth, webdriver flag) while keeping patched getters
# looking native to Function.prototype.toString.
STEALTH_INIT_JS = r"""(() => {
  if (globalThis.__crDesktopProfileApplied) return;
  Object.defineProperty(globalThis, '__crDesktopProfileApplied', {value: true, enumerable: false});

  const nativeStrings = new WeakMap();
  const origToString = Function.prototype.toString;
  const markNative = (fn, name) => {
    nativeStrings.set(fn, `function ${name || ''}() { [native code] }`);
    return fn;
  };
  try {
    const wrappedToString = function toString() {
      if (nativeStrings.has(this)) return nativeStrings.get(this);
      return origToString.call(this);
    };
    nativeStrings.set(wrappedToString, 'function toString() { [native code] }');
    Object.defineProperty(Function.prototype, 'toString', {
      value: wrappedToString, configurable: true, writable: true,
    });
  } catch (_) {}

  const defineGetter = (obj, prop, getter) => {
    try {
      markNative(getter, `get ${prop}`);
      Object.defineProperty(obj, prop, {get: getter, configurable: true});
    } catch (_) {}
  };

  // 1. Ensure navigator.webdriver is false (standard non-automated Chrome value).
  try {
    if (navigator.webdriver !== false) {
      defineGetter(Navigator.prototype, 'webdriver', function webdriver() { return false; });
    }
  } catch (_) {}

  // 2. Consistent desktop languages and Apple Silicon Mac hardware profile.
  try {
    if (!navigator.languages || navigator.languages.length === 0 || navigator.languages[0] === 'C') {
      const langs = Object.freeze(['en-US', 'en']);
      defineGetter(Navigator.prototype, 'languages', function languages() { return langs; });
    }
    if ((navigator.hardwareConcurrency || 0) < 8) {
      defineGetter(Navigator.prototype, 'hardwareConcurrency', function hardwareConcurrency() { return 8; });
    }
    if ((navigator.deviceMemory || 0) < 8) {
      defineGetter(Navigator.prototype, 'deviceMemory', function deviceMemory() { return 8; });
    }
  } catch (_) {}

  // 3. Ensure standard desktop Chrome PDF plugins exist if stripped (e.g. headless mode).
  try {
    if (!navigator.plugins || navigator.plugins.length === 0) {
      const pdfMime = {type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format'};
      const names = [
        'PDF Viewer', 'Chrome PDF Viewer', 'Chromium PDF Viewer',
        'Microsoft Edge PDF Viewer', 'WebKit built-in PDF',
      ];
      const plugins = names.map((name) => {
        const p = {name, filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1, 0: pdfMime};
        return p;
      });
      plugins.item = markNative((i) => plugins[i] || null, 'item');
      plugins.namedItem = markNative((n) => plugins.find((p) => p.name === n) || null, 'namedItem');
      plugins.refresh = markNative(() => {}, 'refresh');
      defineGetter(Navigator.prototype, 'plugins', function pluginsGetter() { return plugins; });
      const mimes = [pdfMime];
      mimes.item = markNative((i) => mimes[i] || null, 'item');
      mimes.namedItem = markNative((t) => mimes.find((m) => m.type === t) || null, 'namedItem');
      defineGetter(Navigator.prototype, 'mimeTypes', function mimeTypesGetter() { return mimes; });
    }
  } catch (_) {}

  // 4. Ensure window.chrome.runtime / app / csi / loadTimes match desktop Google Chrome.
  try {
    if (!window.chrome) window.chrome = {};
    if (!window.chrome.app) {
      window.chrome.app = {
        isInstalled: false,
        InstallState: {DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed'},
        RunningState: {CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running'},
        getDetails: markNative(() => null, 'getDetails'),
        getIsInstalled: markNative(() => false, 'getIsInstalled'),
      };
    }
    if (!window.chrome.csi) {
      window.chrome.csi = markNative(() => ({
        startE: Date.now(), onloadT: Date.now(), pageT: performance.now(), tran: 15,
      }), 'csi');
    }
    if (!window.chrome.loadTimes) {
      window.chrome.loadTimes = markNative(() => ({
        requestTime: Date.now() / 1000, startLoadTime: Date.now() / 1000,
        commitLoadTime: Date.now() / 1000, finishDocumentLoadTime: Date.now() / 1000,
        finishLoadTime: Date.now() / 1000, firstPaintTime: Date.now() / 1000,
        firstPaintAfterLoadTime: 0, navigationType: 'Other', wasFetchedViaSpdy: true,
        wasNpnNegotiated: true, npnNegotiatedProtocol: 'h2', wasAlternateProtocolAvailable: false,
        connectionInfo: 'h2',
      }), 'loadTimes');
    }
    if (!window.chrome.runtime) {
      window.chrome.runtime = {
        OnInstalledReason: {CHROME_UPDATE: 'chrome_update', INSTALL: 'install', SHARED_MODULE_UPDATE: 'shared_module_update', UPDATE: 'update'},
        OnRestartRequiredReason: {APP_UPDATE: 'app_update', OS_UPDATE: 'os_update', PERIODIC: 'periodic'},
        PlatformArch: {ARM: 'arm', ARM64: 'arm64', MIPS: 'mips', MIPS64: 'mips64', X86_32: 'x86-32', X86_64: 'x86-64'},
        PlatformNaclArch: {ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64', X86_32: 'x86-32', X86_64: 'x86-64'},
        PlatformOs: {ANDROID: 'android', CROS: 'cros', FUCHSIA: 'fuchsia', LINUX: 'linux', MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win'},
        RequestUpdateCheckStatus: {NO_UPDATE: 'no_update', THROTTLED: 'throttled', UPDATE_AVAILABLE: 'update_available'},
      };
    }
  } catch (_) {}

  // 5. Mask software VM WebGL renderers (SwiftShader / llvmpipe / Mesa) to match Apple Metal on macOS.
  try {
    const SOFTWARE_RE = /swiftshader|llvmpipe|mesa|subzero|softpipe|software/i;
    const patchWebGL = (Ctor) => {
      if (!Ctor || !Ctor.prototype || typeof Ctor.prototype.getParameter !== 'function') return;
      const orig = Ctor.prototype.getParameter;
      const wrapped = function getParameter(param) {
        const val = orig.call(this, param);
        if (param === 37445 || param === 37446) {
          const renderer = param === 37446 ? String(val || '') : String(orig.call(this, 37446) || '');
          if (SOFTWARE_RE.test(renderer) || SOFTWARE_RE.test(String(val || ''))) {
            return param === 37445
              ? 'Google Inc. (Apple)'
              : 'ANGLE (Apple, ANGLE Metal Renderer: Apple M3, Unspecified Version)';
          }
        }
        return val;
      };
      markNative(wrapped, 'getParameter');
      Object.defineProperty(Ctor.prototype, 'getParameter', {value: wrapped, configurable: true, writable: true});
    };
    patchWebGL(globalThis.WebGLRenderingContext);
    patchWebGL(globalThis.WebGL2RenderingContext);
  } catch (_) {}

  // 6. Non-zero outer window dimensions (headless/minimized windows sometimes report 0).
  try {
    if (window.outerWidth === 0 && window.innerWidth > 0) {
      defineGetter(window, 'outerWidth', function outerWidth() { return window.innerWidth; });
    }
    if (window.outerHeight === 0 && window.innerHeight > 0) {
      defineGetter(window, 'outerHeight', function outerHeight() { return window.innerHeight + 85; });
    }
  } catch (_) {}
})();
"""
