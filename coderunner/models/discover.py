"""Find the best model for this machine on Hugging Face, not only in the curated catalog.

The catalog lists models we have checked ourselves, with measured speed factors. Discovery adds recent
releases from trusted publishers, the same way on Apple silicon and on NVIDIA:

1. list recent chat and vision-chat models from the publishers below, keeping permissive licences;
2. read each model's config.json for what its context cache costs (full-attention layers, KV heads or
   compressed-attention width, sliding windows) and, for mixture-of-experts models, how many experts run
   per token;
3. list its quantized builds with their real file sizes: GGUF repos (llama.cpp, and Ollama through
   hf.co/<repo>:<quant>) and MLX repos on Apple silicon;
4. for this machine, pick the highest-quality build that fits the memory budget at a usable context.

Entries come out in the catalog's format, marked "discovered", and setup must see them pass the
on-device smoke test before using one. Results are cached for a day; without network the catalog alone
is used.
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import date
from typing import Any, Callable

from .. import paths
from .hub import Hub, HubError, endpoint

PUBLISHERS = ("Qwen", "google", "deepseek-ai", "zai-org", "mistralai", "microsoft", "nvidia", "ibm-granite",
              "openai", "prism-ml", "XiaomiMiMo", "moonshotai", "MiniMaxAI", "tencent", "LGAI-EXAONE", "allenai",
              "Hcompany", "stepfun-ai", "inclusionAI", "CohereLabs", "meta-llama", "LiquidAI")
GGUF_PUBLISHERS = ("unsloth", "bartowski", "ggml-org", "lmstudio-community")
CHAT_PIPELINES = {"text-generation", "image-text-to-text", "any-to-any"}
NOT_CHAT = re.compile(r"(?i)(?:^|[-_])(base|embed\w*|rerank\w*|guard|reward|tts|asr|ocr|speech|vae|diffusion|"
                      r"image-\d|draft|eagle|mtp|dspark|probe|singprobe|fp8|bf16|awq|gptq|int4|int8|nvfp4)(?:$|[-_])")
# GGUF files that ride along with a model but are not it: speculative-decoding heads, encoders...
AUXILIARY = re.compile(r"(?i)(mmproj|mtp|draft|eagle|dflash|vision|encoder|audio|tokenizer|projector)")
# Bits per weight of common GGUF quantizations; higher keeps more of the model's quality.
QUANT_BITS = {"BF16": 16, "F16": 16, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "UD-Q5_K_XL": 5.9, "Q5_K_S": 5.5,
              "UD-Q4_K_XL": 4.9, "Q4_K_M": 4.8, "Q4_K_S": 4.6, "MXFP4": 4.25, "IQ4_NL": 4.5, "IQ4_XS": 4.3,
              "UD-Q3_K_XL": 3.9, "Q3_K_M": 3.9, "Q3_K_L": 4.1, "Q3_K_S": 3.5, "IQ3_M": 3.7, "IQ3_XXS": 3.1,
              "UD-Q2_K_XL": 2.9, "Q2_K": 2.6, "IQ2_M": 2.7, "IQ2_XXS": 2.1, "UD-IQ1_S": 1.8, "IQ1_S": 1.6}
QUANT = re.compile(r"(?i)(?:^|[-_./])((?:UD-)?(?:IQ\d_(?:XXS|XS|NL|S|M)|Q\d_K(?:_(?:XL|S|M|L))?|Q\d_\d|BF16|F16|MXFP4))"
                   r"(?=[-_.](?:\d{5}-of|gguf)|\.gguf|/)")
MIN_BITS = 2.0            # below this quality collapses
GOOD_BITS = 4.0           # prefer at least this when it fits
CACHE_TTL = 24 * 3600
RECENT_DAYS = 120


def _get(hub: Hub, url: str, key: str) -> Any:
    return hub._get_json(url, key)


def _api(path: str) -> str:
    return f"{endpoint()}/api/{path}"


# ---- reading a model --------------------------------------------------------------------------------

def arch_from_config(config: dict) -> dict:
    """Context-cache layout from a transformers config.json (text part for vision models)."""
    text = config.get("text_config") or config.get("llm_config") or config
    layers = int(text.get("num_hidden_layers") or 0)
    kinds = text.get("layer_types") or []
    if kinds:
        full = sum(1 for kind in kinds if "full" in str(kind) or kind == "attention")
        sliding = sum(1 for kind in kinds if "sliding" in str(kind))
        linear = sum(1 for kind in kinds if "linear" in str(kind) or "mamba" in str(kind) or "ssm" in str(kind))
        if full == 0 and sliding == 0:     # unfamiliar names: count every non-linear layer as full attention
            full = len(kinds) - linear
    elif text.get("full_attention_interval"):
        full = layers // int(text["full_attention_interval"])
        sliding = 0
    else:
        full, sliding = layers, 0
    heads = int(text.get("num_attention_heads") or 1)
    kv_heads = int(text.get("num_key_value_heads") or heads)
    head_dim = int(text.get("head_dim") or (int(text.get("hidden_size") or 0) // max(heads, 1)))
    arch: dict[str, Any] = {"full_attn_layers": full, "kv_heads": kv_heads, "head_dim": head_dim}
    if text.get("kv_lora_rank"):                       # DeepSeek-style compressed attention (MLA)
        width = int(text["kv_lora_rank"]) + int(text.get("qk_rope_head_dim") or 64)
        arch.update(kv_heads=1, head_dim=max(1, width // 2))   # K and V share one latent; kv_gib counts both
    if sliding and text.get("sliding_window"):
        arch.update(full_attn_layers=full, swa_layers=sliding, swa_kv_heads=kv_heads, swa_head_dim=head_dim,
                    swa_window=int(text["sliding_window"]))
    return arch


def active_fraction(config: dict) -> float:
    """Share of weights used per token: 1 for dense models, roughly experts used / experts for MoE."""
    text = config.get("text_config") or config
    experts = text.get("num_experts") or text.get("n_routed_experts") or text.get("num_local_experts")
    used = text.get("num_experts_per_tok") or text.get("num_experts_per_token") or text.get("moe_topk")
    if not experts or not used:
        return 1.0
    shared = float(text.get("n_shared_experts") or text.get("num_shared_experts") or 0)
    # Attention and embeddings (about 10% of an MoE model) run for every token.
    return min(1.0, 0.1 + 0.9 * (float(used) + shared) / (float(experts) + shared))


def quant_of(filename: str) -> str | None:
    match = QUANT.search(filename)
    if not match:
        return None
    label = match.group(1).upper()
    return label if label in QUANT_BITS else None


def builds(tree: list[dict], params_b: float = 0.0) -> tuple[dict[str, dict], dict | None]:
    """GGUF builds in a repo tree: {quant: {file (first shard), size_gb}}, plus the vision projector.

    A build much smaller than the parameter count implies is not the model (an auxiliary file sharing
    the quant label) and is dropped.
    """
    found: dict[str, dict] = {}
    mmproj = None
    for item in tree:
        name = item.get("path", "")
        if not name.endswith(".gguf"):
            continue
        size = (item.get("lfs") or {}).get("size") or item.get("size") or 0
        if "mmproj" in name.lower():
            if mmproj is None or "q8" in name.lower() or "f16" in name.lower():
                mmproj = {"file": name, "size_gb": size / 1e9}
            continue
        quant = quant_of(name)
        if not quant or AUXILIARY.search(name.rsplit("/", 1)[-1]):
            continue
        entry = found.setdefault(quant, {"file": None, "size_gb": 0.0})
        entry["size_gb"] += size / 1e9
        if entry["file"] is None or "00001-of" in name:
            entry["file"] = name
    if params_b:
        found = {quant: build for quant, build in found.items()
                 if build["size_gb"] >= 0.6 * params_b * QUANT_BITS[quant] / 8}
    return found, mmproj


# ---- searching ---------------------------------------------------------------------------------------

class Discovery:
    def __init__(self, hub: Hub | None = None, since_days: int = RECENT_DAYS,
                 progress: Callable[[str], None] | None = None):
        self.hub = hub or Hub()
        self.since_days = since_days
        self.progress = progress or (lambda message: None)

    def recent(self, publisher: str) -> list[dict]:
        rows = _get(self.hub, _api(f"models?author={publisher}&sort=createdAt&direction=-1&limit=60&full=true"),
                    f"recent:{publisher}:{date.today()}")
        cutoff = time.time() - self.since_days * 86400
        out = []
        for row in rows or []:
            created = row.get("createdAt") or ""
            try:
                stamp = time.mktime(time.strptime(created[:10], "%Y-%m-%d"))
            except ValueError:
                continue
            name = row["id"].split("/", 1)[1]
            if stamp < cutoff or row.get("pipeline_tag") not in CHAT_PIPELINES or NOT_CHAT.search(name):
                continue
            out.append(row)
        return out

    def gguf_repo(self, repo: str) -> tuple[str, list[dict]] | None:
        """The model's own -GGUF repo, else one from a well-known quantizer, with its file tree."""
        name = repo.split("/", 1)[1]
        owner = repo.split("/", 1)[0]
        candidates = [f"{owner}/{name}-GGUF"] + [f"{q}/{name}-GGUF" for q in GGUF_PUBLISHERS] + \
                     [f"bartowski/{owner}_{name}-GGUF"]
        for candidate in candidates:
            try:
                tree = _get(self.hub, _api(f"models/{candidate}/tree/main?recursive=true"), f"tree:{candidate}")
            except HubError:
                continue
            if isinstance(tree, list) and any(str(item.get("path", "")).endswith(".gguf") for item in tree):
                return candidate, tree
        return None

    def mlx_repo(self, repo: str) -> dict | None:
        name = repo.split("/", 1)[1]
        for suffix in ("4bit", "mxfp4", "6bit", "8bit", "3bit"):
            candidate = f"mlx-community/{name}-{suffix}"
            try:
                tree = _get(self.hub, _api(f"models/{candidate}/tree/main?recursive=true"), f"tree:{candidate}")
            except HubError:
                continue
            weights = [item for item in tree or [] if str(item.get("path", "")).endswith(".safetensors")]
            if weights:
                size = sum((item.get("lfs") or {}).get("size") or item.get("size") or 0 for item in weights)
                return {"repo": candidate, "size_gb": round(size / 1e9, 2), "bits": suffix}
        return None

    def describe(self, row: dict) -> dict | None:
        """A catalog-style entry with every GGUF quant listed, or None when there is no usable build."""
        repo = row["id"]
        try:
            info = self.hub.model_info(repo)
            config = _get(self.hub, f"{endpoint()}/{repo}/resolve/main/config.json", f"config:{repo}")
        except HubError:
            return None
        total = ((info.get("safetensors") or {}).get("total") or 0) / 1e9
        if not total or not isinstance(config, dict):
            return None
        license_info = self.hub.resolve_license(repo)
        found = self.gguf_repo(repo)
        quants, mmproj = builds(found[1], total) if found else ({}, None)
        mlx = self.mlx_repo(repo)
        if not quants and not mlx:
            return None
        vision = bool(mmproj) or row.get("pipeline_tag") == "image-text-to-text"
        text = config.get("text_config") or config
        return {
            # A name runtimes accept (Ollama model names allow lowercase letters, digits, ".", "-" and "_").
            "id": "hf-" + re.sub(r"[^a-z0-9._-]+", "-", repo.lower()), "repo": repo,
            "family": str(config.get("model_type") or ""),
            "roles": ["chat"] + (["vision"] if vision else []),
            "params_b": round(total, 2), "active_params_b": round(total * active_fraction(config), 2),
            "arch": arch_from_config(config),
            "ctx_max": int(text.get("max_position_embeddings") or 32768),
            "license": {"spdx": license_info.get("spdx") or "unknown", "base_repo": repo,
                        "link": license_info.get("link"), "commercial": None},
            "released": (row.get("createdAt") or "")[:10], "downloads": row.get("downloads", 0),
            "gguf_repo": found[0] if found else None, "quants": quants, "mmproj": mmproj, "mlx": mlx,
            "discovered": True,
        }

    def run(self, publishers: tuple[str, ...] = PUBLISHERS) -> list[dict]:
        models = []
        for publisher in publishers:
            try:
                rows = self.recent(publisher)
            except HubError as exc:
                self.progress(f"skipping {publisher}: {exc}")
                continue
            for row in rows:
                entry = self.describe(row)
                if entry:
                    models.append(entry)
                    self.progress(f"found {entry['repo']} ({entry['params_b']}B, {len(entry['quants'])} GGUF builds)")
        return models


# ---- fitting to this machine -----------------------------------------------------------------------

def best_build(model: dict, hw, config, runtime: str | None) -> dict | None:
    """The model as a catalog entry with the best-quality build that fits this machine, or None."""
    from . import fit
    options: list[tuple[float, dict]] = []
    if runtime in (None, "llama.cpp", "ollama"):
        for quant, build in model["quants"].items():
            bits = QUANT_BITS.get(quant, 4.0)
            if bits < MIN_BITS:
                continue
            art = {"repo": model["gguf_repo"], "quant": quant, "file": build["file"], "size_gb": round(build["size_gb"], 2),
                   "mmproj_gb": round((model["mmproj"] or {}).get("size_gb", 0.0), 2), "bits": bits}
            artifacts = {"gguf": art, "ollama": {**art, "tag": f"hf.co/{model['gguf_repo']}:{quant}"}}
            options.append((bits, artifacts))
    if runtime in (None, "mlx") and model.get("mlx") and hw.gpu == "apple":
        mlx = model["mlx"]
        bits = {"4bit": 4.5, "mxfp4": 4.25, "6bit": 6.5, "8bit": 8.5, "3bit": 3.5}.get(mlx["bits"], 4.5)
        options.append((bits, {"mlx": {"repo": mlx["repo"], "size_gb": mlx["size_gb"], "bits": bits}}))
    best = None
    for bits, artifacts in sorted(options, key=lambda option: -option[0]):
        entry = {**{k: v for k, v in model.items() if k not in ("quants", "mmproj", "mlx")},
                 "artifacts": artifacts, "rank": 0}
        cand = fit.candidate(entry, hw, config, "chat", runtime)
        if not cand["ok"]:
            continue
        score = (cand["fit"] == "good", bits >= GOOD_BITS, bits)
        if best is None or score > best[0]:
            best = (score, entry, cand)
        if cand["fit"] == "good" and bits >= GOOD_BITS:
            break                      # the highest-quality build that fits well
    return best[1] if best else None


def quality(entry: dict) -> float:
    """A rough prior for ranking models we have not measured: size, active size, recency and precision."""
    total = max(entry.get("params_b", 1), 0.5)
    active = max(entry.get("active_params_b", total), 0.5)
    released = entry.get("released") or "2025-01-01"
    months = (date.fromisoformat(released[:10]) - date(2025, 1, 1)).days / 30 if released[:4].isdigit() else 0
    art = next(iter(entry.get("artifacts", {}).values()), {})
    bits = art.get("bits", 4.8)
    precision = -1.5 if bits < 3 else -0.6 if bits < 4 else 0.0
    return math.log2(total) + 0.5 * math.log2(active) + 0.08 * months + precision


def place(discovered: list[dict], catalog_entries: list[dict]) -> list[dict]:
    """Give discovered entries a rank on the catalog's scale: just behind the curated model whose prior
    is closest below theirs, so measured, curated models win ties."""
    scored = sorted(((quality(e), e["rank"]) for e in catalog_entries if "chat" in e.get("roles", [])), reverse=True)
    for entry in discovered:
        mine = quality(entry)
        below = [rank for score, rank in scored if score <= mine]
        entry["rank"] = (min(below) + 0.5) if below else (max((r for _, r in scored), default=10) + 1)
    return discovered


def cached(progress: Callable[[str], None] | None = None, refresh: bool = False, hub: Hub | None = None) -> list[dict]:
    path = paths.assistant_dir() / "models" / "discovered.json"
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    try:
        models = Discovery(hub, progress=progress).run()
    except Exception as exc:          # offline or rate-limited: the catalog still works
        if progress:
            progress(f"model discovery unavailable: {exc}")
        return json.loads(path.read_text()) if path.exists() else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, indent=1))
    return models
