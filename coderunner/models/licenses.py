"""License policy for downloaded weights.

Permissive licenses are allowed, licenses with use restrictions are accepted
once by the owner (recorded in models.accepted_licenses), and non-commercial,
research-only or unknown licenses are blocked unless the owner lists them in
models.allow_licenses (or accepts that specific repo).
"""

from __future__ import annotations

from .hub import Hub, token

ALLOW = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "unlicense"}
ASK = ("gemma", "llama", "openrail", "creativeml-openrail", "bigscience-openrail", "bigcode-openrail",
       "nvidia-open", "qwen-community", "minimax-community", "tencent-hy", "openmdw", "lfm")
BLOCK = ("cc-by-nc", "research", "apple-amlr", "non-commercial", "noncommercial")


def classify(spdx: str | None, license_name: str | None = None) -> str:
    spdx, name = (spdx or "").strip().lower(), (license_name or "").strip().lower()
    if any(marker in text for text in (spdx, name) for marker in BLOCK):
        return "block"
    if spdx in ALLOW:
        return "allow"
    if spdx.startswith(ASK) or (spdx == "other" and name):
        return "ask"
    return "block"


def key(spdx: str | None, license_name: str | None = None) -> str:
    """The value recorded in models.accepted_licenses for this license."""
    spdx = (spdx or "").strip().lower()
    return (license_name or "").strip().lower() if spdx == "other" and license_name else spdx


def _info(entry_or_repo, hub: Hub | None) -> dict:
    if isinstance(entry_or_repo, str):
        return (hub or Hub()).resolve_license(entry_or_repo)
    lic = entry_or_repo.get("license") or {}
    info = {"spdx": lic.get("spdx"), "name": lic.get("name"), "link": lic.get("link"),
            "repo": lic.get("base_repo") or entry_or_repo.get("id"), "gated": bool(lic.get("gated"))}
    if not info["spdx"] and lic.get("base_repo"):
        info = (hub or Hub()).resolve_license(lic["base_repo"])
    return info


def check(entry_or_repo, config, hub: Hub | None = None) -> tuple[str, str, str]:
    """(decision, reason, link) for a catalog entry or a Hugging Face repo id."""
    info = _info(entry_or_repo, hub)
    spdx, name, link = info.get("spdx"), info.get("name"), info.get("link") or ""
    label = name or spdx or "no license"
    k = key(spdx, name)
    allowed = {str(x).lower() for x in config.get("models.allow_licenses") or []}
    accepted = {str(x).lower() for x in config.get("models.accepted_licenses") or []}
    gated = " (gated on Hugging Face: accept the terms there and set HF_TOKEN)" if info.get("gated") and not token() else ""
    verdict = classify(spdx, name)

    if verdict == "allow":
        return "allow", f"{label} permits commercial use{gated}", link
    if k and k in allowed:
        return "allow", f"{label} is listed in models.allow_licenses{gated}", link
    if verdict == "ask":
        if k in accepted:
            return "allow", f"{label} was accepted by the owner{gated}", link
        return "ask", f"{label} has use restrictions; review {link} and accept it once{gated}", link
    repo = str(info.get("repo") or "").lower()
    if not spdx and repo and repo in accepted:
        return "allow", f"{info['repo']} has no license metadata but was accepted by the owner", link
    if not spdx:
        return "block", f"{info.get('repo') or 'this model'} has no license metadata", link
    if any(marker in f"{spdx} {name or ''}".lower() for marker in BLOCK):
        return "block", f"{label} does not allow commercial use", link
    return "block", f"{label} is not a recognised license; list it in models.allow_licenses to use it", link
