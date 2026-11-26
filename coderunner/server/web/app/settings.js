// macOS-style Settings sheet (Cmd+, or sidebar footer): seven quiet sections using CodeRunner's warm paper /
// charcoal palette, hairline inset lists, and plain English. Every control talks directly to the owner API.
import {el} from "./render.js";

const ICONS = {
  models: "M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2 M7 7h10v10H7z",
  files: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  apps: "M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z",
  browser: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18 M3 12h18 M12 3a14 14 0 0 1 0 18 M12 3a14 14 0 0 0 0 18",
  routines: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18 M12 7v5l3 2",
  privacy: "M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6z",
  voice: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z M6 11a6 6 0 0 0 12 0 M12 17v4",
};

const SECTIONS = [
  {id: "models", label: "Models & Mac"},
  {id: "files", label: "Files & Search"},
  {id: "apps", label: "Apps & Access"},
  {id: "browser", label: "Browser"},
  {id: "routines", label: "Routines"},
  {id: "privacy", label: "Privacy & Safety"},
  {id: "voice", label: "Voice & Phone"},
];

function cfgGet(cfgData, key, fallback = undefined) {
  const flat = cfgData && (cfgData.settings || cfgData.values);
  if (flat && Object.prototype.hasOwnProperty.call(flat, key) && flat[key] !== undefined) return flat[key];
  let cur = cfgData && cfgData.config;
  for (const part of String(key).split(".")) {
    if (!cur || typeof cur !== "object") { cur = undefined; break; }
    cur = cur[part];
  }
  return cur !== undefined ? cur : fallback;
}

function svgIcon(d) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("ico");
  const p = document.createElementNS(svg.namespaceURI, "path");
  p.setAttribute("d", d);
  svg.appendChild(p);
  return svg;
}

function fmtGb(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "";
  return n >= 10 ? `${Math.round(n)} GB` : `${n.toFixed(1)} GB`;
}

function toggleSwitch(checked, onChange, ariaLabel) {
  const wrap = el("label", "switch");
  const input = el("input");
  input.type = "checkbox";
  input.checked = Boolean(checked);
  if (ariaLabel) input.setAttribute("aria-label", ariaLabel);
  input.onchange = () => onChange(input.checked, input);
  wrap.append(input, el("span", "switch-track"));
  return wrap;
}

function group(title, rows, noteText) {
  const sec = el("div", "set-group");
  if (title) sec.appendChild(el("div", "set-group-title", title));
  const list = el("div", "set-list");
  for (const r of rows) if (r) list.appendChild(r);
  sec.appendChild(list);
  if (noteText) sec.appendChild(el("p", "set-note muted small", noteText));
  return sec;
}

function row(label, desc, control) {
  const r = el("div", "set-row");
  const left = el("div", "set-row-text");
  left.appendChild(el("div", "set-row-label", label));
  if (desc) left.appendChild(el("div", "set-row-desc muted small", desc));
  const right = el("div", "set-row-ctrl");
  if (Array.isArray(control)) {
    for (const c of control) if (c) right.appendChild(c);
  } else if (control) {
    right.appendChild(control);
  }
  r.append(left, right);
  return r;
}

export function createSettings({api, modelName, notice, onRestart, onRunCheck, onReloadModels, onOpenMacBrowser}) {
  let backdrop = null;
  let active = "models";
  let restartNeeded = false;
  let restartBar = null;
  let paneBody = null;
  let railButtons = new Map();
  let benchPoll = null;

  function markRestart() {
    restartNeeded = true;
    if (restartBar) restartBar.hidden = false;
  }

  function close() {
    clearTimeout(benchPoll);
    if (backdrop) { backdrop.remove(); backdrop = null; }
  }

  async function saveSetting(patch) {
    const res = await api("/api/settings", {method: "PATCH", body: JSON.stringify(patch)});
    if (res.restart_required) markRestart();
    return res;
  }

  function buildShell() {
    backdrop = el("div", "set-backdrop");
    const sheet = el("div", "set-sheet");
    sheet.setAttribute("role", "dialog");
    sheet.setAttribute("aria-label", "Settings");

    const rail = el("nav", "set-rail");
    rail.setAttribute("aria-label", "Settings sections");
    rail.appendChild(el("div", "set-rail-head", "Settings"));
    railButtons = new Map();
    for (const s of SECTIONS) {
      const btn = el("button", "set-nav");
      btn.type = "button";
      btn.append(svgIcon(ICONS[s.id]), el("span", "", s.label));
      btn.onclick = () => select(s.id);
      rail.appendChild(btn);
      railButtons.set(s.id, btn);
    }

    const right = el("div", "set-pane");
    const top = el("div", "set-pane-head");
    const title = el("h2", "set-pane-title");
    const closeBtn = el("button", "icon-btn set-close");
    closeBtn.appendChild(svgIcon("M6 6l12 12 M18 6l-12 12"));
    closeBtn.type = "button";
    closeBtn.title = "Close (Esc)";
    closeBtn.setAttribute("aria-label", "Close settings");
    closeBtn.onclick = close;
    top.append(title, closeBtn);

    paneBody = el("div", "set-pane-body");

    restartBar = el("div", "set-restart");
    restartBar.hidden = !restartNeeded;
    const rBtn = el("button", "btn primary", "Restart now");
    rBtn.type = "button";
    rBtn.onclick = async () => {
      rBtn.disabled = true;
      rBtn.textContent = "Restarting…";
      try {
        await api("/api/restart", {method: "POST", body: "{}"});
        close();
        onRestart?.();
      } catch (err) {
        rBtn.disabled = false;
        rBtn.textContent = "Restart now";
        notice(err.message);
      }
    };
    restartBar.append(el("span", "", "Restart the assistant to apply these changes."), rBtn);

    right.append(top, paneBody, restartBar);
    sheet.append(rail, right);
    backdrop.appendChild(sheet);
    backdrop.onclick = (e) => { if (e.target === backdrop) close(); };
    document.body.appendChild(backdrop);
    return {title};
  }

  let titleNode = null;
  async function select(sectionId) {
    clearTimeout(benchPoll);
    const scrollToBench = sectionId === "bench";
    if (scrollToBench) sectionId = "models";
    active = SECTIONS.some((s) => s.id === sectionId) ? sectionId : "models";
    for (const [id, btn] of railButtons) btn.classList.toggle("on", id === active);
    const meta = SECTIONS.find((s) => s.id === active);
    if (titleNode) titleNode.textContent = meta ? meta.label : "Settings";
    paneBody.replaceChildren(el("p", "muted small", "Loading…"));
    try {
      if (active === "models") {
        await renderModels(paneBody);
        if (scrollToBench) paneBody.querySelector("#benchLabSection")?.scrollIntoView({block: "start"});
      }
      else if (active === "files") await renderFiles(paneBody);
      else if (active === "apps") await renderApps(paneBody);
      else if (active === "browser") await renderBrowser(paneBody);
      else if (active === "routines") await renderRoutines(paneBody);
      else if (active === "privacy") await renderPrivacy(paneBody);
      else if (active === "voice") await renderVoice(paneBody);
    } catch (err) {
      paneBody.replaceChildren(el("p", "setup-error", err.message || "Could not load this section."));
    }
  }

  // ---- 1. Models, Hardware, Hugging Face & Benchmark Lab --------------------------------------------
  async function renderModels(body) {
    const [data, remote, benchData, cfgData] = await Promise.all([
      api("/api/models"), api("/api/remote"), api("/api/bench"), api("/api/settings"),
    ]);
    if (active !== "models") return;
    body.replaceChildren();
    const hw = data.hardware || {}, mem = data.live_memory || {}, probe = data.probe || benchData.probe;
    const plan = data.plan || {}, tier = hw.tier || plan.tier;

    // Hardware & ANE strip (4 calm tiles, tabular numbers)
    const tiles = el("div", "tiles set-tiles");
    const tile = (val, lbl, hint) => {
      const box = el("div", "tile");
      box.append(el("div", "tile-value", val), el("div", "tile-label", lbl));
      if (hint) box.title = hint;
      tiles.appendChild(box);
    };
    tile(hw.chip || "This computer", tier ? `${tier.label} unified RAM` : "Chip");
    const freeTxt = mem.free_gb != null ? `${fmtGb(mem.free_gb)} free now` : `${fmtGb(hw.budget_gb)} for models`;
    const swapNote = mem.swap_used_gb >= 1.0 ? ` · ${fmtGb(mem.swap_used_gb)} swap` : "";
    tile(freeTxt, `Memory (${mem.pressure || "ok"}${swapNote})`, `${fmtGb(hw.budget_gb)} model budget of ${hw.ram_gb} GB`);
    if (probe && probe.ok) {
      tile(`${Math.round(probe.bandwidth_gbs || hw.bandwidth_gbs)} GB/s`,
           probe.gpu_fp16_tflops ? `GPU · ${probe.gpu_fp16_tflops} TFLOPS` : "GPU bandwidth",
           `Measured ${probe.date || ""}`);
      tile(probe.ane_verified ? `${probe.ane_cores}-core ANE` : `${probe.ane_cores || 0} cores`,
           probe.ane_verified
             ? `${probe.ane_fp16_tops || "—"} TOPS${probe.gpu_contention_pct != null ? ` · ${probe.gpu_contention_pct}% GPU` : ""}`
             : "Neural Engine",
           probe.ane_plan_cost_pct != null ? `MLComputePlan: ${probe.ane_plan_cost_pct}% on Neural Engine` : "");
    } else {
      tile(`${Math.round(hw.bandwidth_gbs || 0)} GB/s`, "Memory speed (est.)", "Run Mac hardware bench to measure directly");
      tile(hw.gpu === "apple" ? "Apple Silicon" : hw.gpu || "CPU", "Neural Engine", "Run Mac hardware bench to test ANE");
    }
    body.appendChild(tiles);

    // Where tasks run: Local / Your server / Claude
    const modeRow = el("div", "set-seg");
    const curMode = remote.on ? (remote.provider === "openai" ? "server" : "claude") : "local";
    let modeView = curMode;
    const modeBox = el("div", "set-mode-box");

    const drawMode = () => {
      modeRow.replaceChildren();
      for (const [key, label] of [["local", "On this Mac"], ["server", "Your server"], ["claude", "Claude"]]) {
        const b = el("button", "set-seg-btn" + (modeView === key ? " on" : ""), label);
        b.type = "button";
        b.onclick = () => { modeView = key; drawMode(); };
        modeRow.appendChild(b);
      }
      modeBox.replaceChildren();
      if (modeView === "local") {
        if (remote.on) {
          const back = el("button", "btn primary", "Switch back to this Mac");
          back.type = "button";
          back.onclick = async () => {
            back.disabled = true;
            await api("/api/remote", {method: "POST", body: JSON.stringify({action: "off"})});
            close(); onRestart?.();
          };
          modeBox.appendChild(group("Currently using a remote model", [row(modelName(remote.model), remote.warning || "", back)]));
        }
        // Tier-aware Fast / Balanced / Smartest presets
        const presets = plan.presets || [];
        if (presets.length) {
          const pGrid = el("div", "preset-grid");
          const curId = (data.active_stack || {}).planner || "";
          let activeMarked = false;
          for (const p of presets) {
            const isOn = !activeMarked && p.model === curId;
            if (isOn) activeMarked = true;
            const card = el("button", "preset-card" + (isOn ? " on" : ""));
            card.type = "button";
            card.append(
              el("div", "preset-head", p.label),
              el("div", "preset-model", modelName(p.model)),
              el("div", "preset-meta muted small",
                 `${p.step_s ? `~${Math.round(p.step_s)} s/step · ` : ""}${fmtGb(p.need_gb)} · ${p.vision ? "Vision" : "Text only"}`),
            );
            card.onclick = async () => {
              if (p.model === curId) return;
              card.disabled = true;
              try {
                await api("/api/models", {method: "POST", body: JSON.stringify({action: "switch", model: p.model})});
                notice(`Switching to ${modelName(p.model)}…`);
                close();
                onReloadModels?.();
              } catch (err) { card.disabled = false; notice(err.message); }
            };
            pGrid.appendChild(card);
          }
          const wrap = el("div", "set-group");
          wrap.append(el("div", "set-group-title", `Recommended for this ${tier ? tier.label : ""} Mac`), pGrid);
          modeBox.appendChild(wrap);
        }

        // Wired memory limit hint on 48-512 GB Macs (quiet disclosure, never dumped raw)
        if (plan.wired_limit_hint) {
          const h = plan.wired_limit_hint;
          const det = el("details", "set-fold");
          det.append(el("summary", "", `Unlock ${h.model_name} (${fmtGb(h.need_gb)}) by raising macOS GPU memory limit to ${fmtGb(h.limit_gb)}`));
          const copyBtn = el("button", "btn", "Copy command");
          copyBtn.type = "button";
          copyBtn.onclick = () => { navigator.clipboard?.writeText(h.command); copyBtn.textContent = "Copied"; };
          const code = el("code", "set-code", h.command);
          const inner = el("div", "set-fold-body");
          inner.append(el("p", "muted small", "macOS caps GPU memory at 75% by default. Running this once in Terminal raises it until restart:"),
                       el("div", "set-code-row"), );
          inner.querySelector(".set-code-row").append(code, copyBtn);
          det.appendChild(inner);
          modeBox.appendChild(det);
        }

        // Specialist roles (Browser driver, Embeddings, Hugging Face discovery, Thinking)
        const drv = (data.active_stack || {}).driver || "";
        const drvSel = el("select", "set-select");
        for (const [val, lbl] of [["", "Same as planning model"], ["fara1.5-4b", "Fara 1.5 4B (web screenshots)"],
                                  ["fara1.5-9b", "Fara 1.5 9B (48 GB+ Macs)"], ["mage-vl-cowork", "Mage VL Cowork 4.7B"]]) {
          const opt = el("option", "", lbl); opt.value = val; opt.selected = drv === val; drvSel.appendChild(opt);
        }
        drvSel.onchange = async () => {
          drvSel.disabled = true;
          try {
            const payload = drvSel.value ? {action: "driver", model: drvSel.value} : {action: "driver", off: true};
            await api("/api/models", {method: "POST", body: JSON.stringify(payload)});
            markRestart();
            notice(drvSel.value ? `Set browser specialist to ${modelName(drvSel.value)}.` : "Browser specialist turned off.");
          } catch (err) { notice(err.message); }
          finally { drvSel.disabled = false; }
        };

        const thinkSel = el("select", "set-select");
        const curThink = String(cfgGet(cfgData, "agent.thinking", "auto") || "auto");
        for (const [val, lbl] of [["auto", "Automatic"], ["on", "Always think first"], ["off", "Off (fastest)"]]) {
          const opt = el("option", "", lbl); opt.value = val; opt.selected = curThink === val; thinkSel.appendChild(opt);
        }
        thinkSel.onchange = () => saveSetting({"agent.thinking": thinkSel.value});

        const hfBtn = el("button", "btn", "Check Hugging Face");
        hfBtn.type = "button";
        hfBtn.onclick = async () => {
          hfBtn.disabled = true; hfBtn.textContent = "Checking…";
          await api("/api/bench", {method: "POST", body: JSON.stringify({action: "discover"})}).catch((e) => notice(e.message));
          watchBench();
        };

        const idleSel = el("select", "set-select");
        const curIdle = Number(cfgGet(cfgData, "models.idle_unload_minutes", 10) ?? 10);
        for (const [val, lbl] of [[5, "5 minutes"], [10, "10 minutes"], [30, "30 minutes"], [0, "Never"]]) {
          const opt = el("option", "", lbl); opt.value = String(val); opt.selected = curIdle === val; idleSel.appendChild(opt);
        }
        idleSel.onchange = () => saveSetting({"models.idle_unload_minutes": Number(idleSel.value)});

        modeBox.appendChild(group("Roles & Hugging Face", [
          row("Web browsing specialist", "Optional vision model trained to navigate complex websites.", drvSel),
          row("Reasoning before steps", "Automatic thinks before the first step and after failures.", thinkSel),
          row("Check Hugging Face for newer models", "Finds newly released Apache/MIT models that fit this Mac.",
              [toggleSwitch(cfgGet(cfgData, "models.discover", true) !== false, (v) => saveSetting({"models.discover": v}), "Discover on Hugging Face"), hfBtn]),
          row("Unload model when idle", "Frees RAM automatically when you are not running tasks.", idleSel),
        ]));
      } else if (modeView === "server") {
        const urlIn = el("input", "set-input"); urlIn.placeholder = "http://100.64.0.5:8080/v1";
        urlIn.value = remote.provider === "openai" ? (remote.base_url || "") : "";
        const modIn = el("input", "set-input"); modIn.placeholder = "Model name (e.g. bonsai2.5-27b)";
        modIn.value = remote.provider === "openai" ? (remote.model || "") : "";
        const keyIn = el("input", "set-input"); keyIn.type = "password"; keyIn.placeholder = "API key (optional)";
        const save = el("button", "btn primary", "Connect server");
        save.type = "button";
        save.onclick = async () => {
          save.disabled = true; save.textContent = "Checking…";
          try {
            await api("/api/remote", {method: "POST", body: JSON.stringify({
              action: "server", base_url: urlIn.value.trim(), model: modIn.value.trim(), key: keyIn.value.trim(),
            })});
            close(); onRestart?.();
          } catch (err) { save.disabled = false; save.textContent = "Connect server"; notice(err.message); }
        };
        modeBox.appendChild(group("Your own OpenAI-compatible server", [
          row("Server URL", "llama-server, vLLM, or LM Studio on another machine or your tailnet.", urlIn),
          row("Model ID", "The exact model name the server answers to.", modIn),
          row("Access key", "Stored locally with 0600 permissions; leave blank if none.", keyIn),
          row("Apply", "File search and voice stay on this Mac.", save),
        ]));
      } else {
        const keyIn = el("input", "set-input"); keyIn.type = "password";
        keyIn.placeholder = remote.has_key ? "Key saved (paste to replace)" : "Anthropic API key (sk-ant-…)";
        const modSel = el("select", "set-select");
        for (const m of (remote.models || [{id: "claude-haiku-5-5"}, {id: "claude-sonnet-5-5"}, {id: "claude-opus-5-5"}])) {
          const opt = el("option", "", modelName(m.id)); opt.value = m.id; opt.selected = remote.model === m.id; modSel.appendChild(opt);
        }
        const go = el("button", "btn primary", remote.on && remote.provider === "anthropic" ? "Update Claude" : "Use Claude");
        go.type = "button";
        go.onclick = async () => {
          go.disabled = true;
          try {
            await api("/api/remote", {method: "POST", body: JSON.stringify({
              action: "on", key: keyIn.value.trim() || undefined, model: modSel.value,
            })});
            close(); onRestart?.();
          } catch (err) { go.disabled = false; notice(err.message); }
        };
        modeBox.appendChild(group("Anthropic Claude (not private)", [
          row("Model", "Sends your messages, opened pages, and attached files to Anthropic.", modSel),
          row("API key", "Kept only in ~/.coderunner/assistant/anthropic-key.", keyIn),
          row("Remote mode", "", go),
        ], remote.warning || "Requests sent to Anthropic are not private."));
      }
    };
    body.append(modeRow, modeBox);
    drawMode();

    // Benchmarks: Hardware + ANE probe, Tier 1 gate, Single vs Multi-Model Comparison
    const lab = el("div", "set-group");
    lab.id = "benchLabSection";
    lab.appendChild(el("div", "set-group-title", "Benchmarks"));
    const labCard = el("div", "set-list");
    const statusLine = el("div", "set-bench-status muted small");
    const probeBtn = el("button", "btn", "Measure hardware");
    probeBtn.type = "button";
    probeBtn.onclick = async () => {
      await api("/api/bench", {method: "POST", body: JSON.stringify({action: "probe", force: true})}).catch((e) => notice(e.message));
      watchBench();
    };
    const cmpBtn = el("button", "btn primary", "Run");
    cmpBtn.type = "button";
    const suiteSel = el("select", "set-select");
    for (const [v, l] of [["tier1", "Quick check (~90s)"], ["tier2", "Sample tasks (~6 min)"], ["ane", "Neural Engine & voice"]]) {
      const o = el("option", "", l); o.value = v; suiteSel.appendChild(o);
    }
    // Select only the currently active stack by default so clicking Run never starts an unexpected multi-GB download.
    const chosenIds = new Set((benchData.candidates || []).slice(0, 1).map((c) => c.id));
    cmpBtn.onclick = async () => {
      const stacks = (benchData.candidates || []).filter((c) => chosenIds.has(c.id));
      if (!stacks.length) { notice("Pick at least one setup to benchmark."); return; }
      await api("/api/bench", {method: "POST", body: JSON.stringify({action: "compare", suite: suiteSel.value, stacks, reuse_cached: false})})
        .catch((e) => notice(e.message));
      watchBench();
    };
    const healthBtn = el("button", "btn", "Test this Mac");
    healthBtn.type = "button";
    healthBtn.onclick = () => { close(); onRunCheck?.(); };

    labCard.appendChild(row("Hardware probe", "Memory bandwidth, Metal FP16, Neural Engine TOPS and GPU/ANE contention.", probeBtn));
    labCard.appendChild(row("Model benchmark", "Test your active model or compare single vs. multi-model stacks.", [suiteSel, cmpBtn]));
    if ((benchData.candidates || []).length) {
      const chips = el("div", "set-chips");
      for (const c of benchData.candidates.slice(0, 5)) {
        const ch = el("button", "set-chip" + (chosenIds.has(c.id) ? " on" : ""), c.label);
        ch.type = "button";
        ch.onclick = () => {
          if (chosenIds.has(c.id)) chosenIds.delete(c.id); else if (chosenIds.size < 4) chosenIds.add(c.id);
          ch.classList.toggle("on", chosenIds.has(c.id));
        };
        chips.appendChild(ch);
      }
      labCard.appendChild(chips);
    }
    labCard.appendChild(row("Install health check", "Runs one of each everyday task type on the active setup.", healthBtn));
    lab.appendChild(labCard);
    if (benchData.state === "failed" && benchData.result?.error) {
      statusLine.textContent = `Last run failed: ${benchData.result.error}`;
      statusLine.style.color = "var(--danger)";
    } else if (benchData.state === "done" && benchData.lines?.length) {
      statusLine.textContent = benchData.lines.slice(-1)[0];
    }
    lab.appendChild(statusLine);

    // Persistent multi-stack scoreboard with expandable check/task diagnostics
    const historyByKey = new Map((benchData.history || []).map((h) => [`${h.signature || ""}|${h.suite || ""}`, h]));
    const rows = (benchData.summary && benchData.summary.rows) || [];
    const fallbackRuns = !rows.length && benchData.result?.runs?.length ? benchData.result.runs : [];
    const displayRuns = rows.length ? rows : fallbackRuns.map((run) => ({
      label: (run.stack && run.stack.label) || run.signature || "Setup",
      passed: run.passed || 0, total: run.total || 0, step_s: run.step_s || 0,
      memory_gb: (run.memory && run.memory.total_gb) || 0, comfort: (run.memory && run.memory.comfort) || "",
      cache_hit_ratio: run.cache_hit_ratio, stack: run.stack, suite: run.suite,
      checks: run.checks, tasks: run.tasks, raw: run,
    }));
    if (displayRuns.length) {
      const board = el("div", "set-list set-board");
      for (const r of displayRuns.slice(0, 6)) {
        const raw = r.raw || historyByKey.get(`${(r.stack && r.stack.id) || ""}|${r.suite || ""}`) ||
          (benchData.history || []).find((h) => (h.stack?.label === r.label) && (!r.suite || h.suite === r.suite)) || {};
        const suiteTag = r.suite === "tier1" ? "Quick" : r.suite === "tier2" ? "Tasks" : r.suite || "";
        const parts = [
          suiteTag,
          `${r.passed}/${r.total} passed`,
          r.step_s ? `~${r.step_s}s/step` : "",
          raw.read_tok_s ? `${Math.round(raw.read_tok_s)} r / ${Math.round(raw.gen_tok_s || 0)} w tok/s` : "",
          r.cache_hit_ratio != null ? `${Math.round(r.cache_hit_ratio * 100)}% cache` : "",
          r.memory_gb ? `${r.memory_gb} GB` : "Server",
        ].filter(Boolean);
        const ctrls = [];
        const items = raw.checks || r.checks || raw.tasks || r.tasks || [];
        let detailBox = null;
        if (items.length) {
          const detBtn = el("button", "linkish small", "Details");
          detBtn.type = "button";
          detBtn.onclick = () => { if (detailBox) detailBox.hidden = !detailBox.hidden; };
          ctrls.push(detBtn);
        }
        if (r.stack && r.stack.planner && !r.stack.remote_provider) {
          const useBtn = el("button", "btn", "Use");
          useBtn.type = "button";
          useBtn.onclick = async () => {
            useBtn.disabled = true;
            await api("/api/models", {method: "POST", body: JSON.stringify({action: "switch", model: r.stack.planner})}).catch((e) => notice(e.message));
            close(); onReloadModels?.();
          };
          ctrls.push(useBtn);
        }
        board.appendChild(row(r.label, parts.join(" · "), ctrls));
        if (items.length) {
          detailBox = el("div", "set-bench-detail");
          detailBox.hidden = r.passed === r.total && r.total > 0;
          for (const it of items) {
            const ok = Boolean(it.passed);
            const line = el("div", "set-bench-item " + (ok ? "ok" : "fail"));
            const left = el("span", "");
            left.append(el("span", "mk", ok ? "✓" : "✕"), el("span", "", it.name || "check"));
            const why = [it.detail || it.why || "", it.seconds != null ? `${it.seconds}s` : ""].filter(Boolean).join(" · ");
            line.append(left, el("span", "set-bench-item-why", why));
            line.title = why;
            detailBox.appendChild(line);
          }
          board.appendChild(detailBox);
        }
      }
      lab.appendChild(board);
    }
    body.appendChild(lab);

    // Storage & downloaded model cleanup
    const changeDir = el("button", "btn", "Change folder…");
    changeDir.type = "button";
    changeDir.onclick = async () => {
      const picked = await window.coderunnerShell?.pickFolder?.({start: data.models_dir, title: "Where to keep models", buttonLabel: "Keep them here"});
      if (!picked) return;
      await api("/api/setup", {method: "POST", body: JSON.stringify({action: "location", path: picked})}).catch((e) => notice(e.message));
      select("models");
    };
    const storageRows = [row("Models folder", data.models_dir || "", changeDir)];
    const WEIGHT_LABELS = {coreml: "Core ML embeddings", speech: "On-device voice models"};
    for (const w of (data.weights || []).filter((x) => x.gb >= 0.05).slice(0, 8)) {
      const rm = el("button", "btn danger", "Remove");
      rm.type = "button";
      rm.onclick = async () => {
        rm.disabled = true;
        await api("/api/models", {method: "POST", body: JSON.stringify({action: "remove_weights", id: w.id})}).catch((e) => notice(e.message));
        select("models");
      };
      const label = WEIGHT_LABELS[w.id] || w.id.replace(/--/g, "/");
      storageRows.push(row(label, `${fmtGb(w.gb)} · ${w.files} ${w.files === 1 ? "file" : "files"}`, rm));
    }
    body.appendChild(group("Storage", storageRows));

    const watchBench = async () => {
      clearTimeout(benchPoll);
      if (active !== "models" || !backdrop) return;
      const st = await api("/api/bench").catch(() => null);
      if (!st) return;
      if (st.state === "running") {
        statusLine.textContent = (st.lines || []).slice(-1)[0] || `Running ${st.kind}…`;
        benchPoll = setTimeout(watchBench, 1200);
      } else if (st.state === "done" || st.state === "failed") {
        if (st.state === "failed" && st.result?.error) notice(st.result.error);
        select("models");
      }
    };
    if (benchData.state === "running") watchBench();
  }

  // ---- 2. Files & Search ---------------------------------------------------------------------------
  async function renderFiles(body) {
    const [files, cfgData] = await Promise.all([api("/api/files"), api("/api/settings")]);
    if (active !== "files") return;
    body.replaceChildren();
    const s = cfgData.settings || {};

    const addBtn = el("button", "btn primary", "Add folder…");
    addBtn.type = "button";
    addBtn.onclick = async () => {
      const picked = await window.coderunnerShell?.pickFolder?.({title: "Share a folder with CodeRunner", buttonLabel: "Share folder"});
      if (!picked) return;
      await api("/api/files", {method: "POST", body: JSON.stringify({add: picked})}).catch((e) => notice(e.message));
      select("files");
    };
    const folderRows = [];
    for (const root of (files.roots || [])) {
      const rm = el("button", "btn", "Remove");
      rm.type = "button";
      rm.onclick = async () => {
        await api("/api/files", {method: "POST", body: JSON.stringify({remove: root})}).catch((e) => notice(e.message));
        select("files");
      };
      folderRows.push(row(root.split("/").pop() || root, root, rm));
    }
    const chips = el("div", "set-chips");
    for (const sug of (cfgData.suggestions || [])) {
      if (sug.added) continue;
      const b = el("button", "set-chip", `+ ${sug.name}`);
      b.type = "button";
      b.onclick = async () => {
        await api("/api/files", {method: "POST", body: JSON.stringify({add: sug.path})}).catch((e) => notice(e.message));
        select("files");
      };
      chips.appendChild(b);
    }
    const totalFiles = files.files != null
      ? Number(files.files)
      : Object.values(files.by_kind || {}).reduce((acc, k) => acc + Number((k && k.files) || 0), 0);
    const idxText = totalFiles > 0 ? `${totalFiles.toLocaleString()} files indexed` : "Background index ready";
    folderRows.push(row("Shared folders", idxText, addBtn));
    const fg = group("Folders on this Mac", folderRows, "Only shared folders are indexed. Keys, .env files, and ~/Library are always skipped.");
    if (chips.childNodes.length) fg.insertBefore(chips, fg.lastChild);
    body.appendChild(fg);

    // Mail / Messages / Notes / Calendar indexing (/api/files returns sources as {chosen, enabled, indexed} or an array)
    const rawSources = files.sources;
    const sourceList = Array.isArray(rawSources)
      ? rawSources
      : [...(rawSources?.chosen || []), ...(rawSources?.enabled || [])];
    const sources = new Set(sourceList);
    const sourceCounts = (rawSources && !Array.isArray(rawSources) && rawSources.indexed) || {};
    const srcRows = [];
    for (const [id, label, desc] of [
      ["mail", "Mail", "Index inbox and sent messages (codes and reset links redacted)."],
      ["messages", "Messages", "Index iMessage conversations (needs Full Disk Access)."],
      ["notes", "Notes", "Search Apple Notes alongside your files."],
      ["calendar", "Calendar", "Index upcoming and recent events."],
    ]) {
      const n = Number(sourceCounts[id] || 0);
      const fullDesc = n > 0 ? `${desc} (${n.toLocaleString()} indexed)` : desc;
      const ctrls = [toggleSwitch(sources.has(id), async (on) => {
        await api("/api/files", {method: "POST", body: JSON.stringify(on ? {source_add: id} : {source_remove: id})})
          .catch((e) => notice(e.message));
        select("files");
      }, label)];
      if (id === "messages" && window.coderunnerShell?.openSystemSettings) {
        const fda = el("button", "linkish small", "Full Disk Access");
        fda.type = "button";
        fda.onclick = () => window.coderunnerShell.openSystemSettings("Privacy_AllFiles");
        ctrls.unshift(fda);
      }
      srcRows.push(row(label, fullDesc, ctrls));
    }
    body.appendChild(group("Apps in search", srcRows));

    body.appendChild(group("Indexing preferences", [
      row("Index photos, audio and video", "Uses Core ML on the Neural Engine to search pictures and recordings.",
          toggleSwitch(s["files.media"] !== false, (v) => saveSetting({"files.media": v}), "Index media")),
      row("Index media on battery", "Off by default so heavy media indexing waits for power.",
          toggleSwitch(Boolean(s["files.media_on_battery"]), (v) => saveSetting({"files.media_on_battery": v}), "Media on battery")),
      row("Pause indexing during chat replies", "Keeps local models responsive when both share unified memory.",
          toggleSwitch(s["files.yield_during_llm_turn"] !== "false",
                       (v) => saveSetting({"files.yield_during_llm_turn": v ? "auto" : "false"}), "Yield indexing")),
      row("OCR scanned PDFs and screenshots", "Reads text inside images and scans on device.",
          toggleSwitch(s["files.ocr"] !== false, (v) => saveSetting({"files.ocr": v}), "OCR")),
      row("Read small iCloud-only files", "Downloads files under 20 MB to index them, then evicts them back to iCloud.",
          toggleSwitch(Number(s["files.cloud_file_mb"] ?? 20) > 0,
                       (v) => saveSetting({"files.cloud_file_mb": v ? 20 : 0}), "iCloud files")),
    ]));
  }

  // ---- 3. Apps & Permissions (macOS Connectors) ----------------------------------------------------
  async function renderApps(body) {
    const data = await api("/api/connectors");
    if (active !== "apps") return;
    body.replaceChildren();
    if (!data.supported) {
      body.appendChild(el("p", "muted", "App connectors are available on macOS."));
      return;
    }
    const rows = [];
    for (const c of (data.connectors || [])) {
      const testOut = el("span", "muted small");
      const testBtn = el("button", "btn", "Test");
      testBtn.type = "button";
      testBtn.onclick = async () => {
        testBtn.disabled = true; testBtn.textContent = "Testing…";
        const res = await api("/api/connectors", {method: "POST", body: JSON.stringify({action: "test", name: c.name})})
          .catch((e) => ({ok: false, error: e.message}));
        testBtn.disabled = false; testBtn.textContent = "Test";
        testOut.textContent = res.ok ? "Works" : (res.error || "Permission needed");
      };
      const sw = toggleSwitch(c.enabled, async (on) => {
        const res = await api("/api/connectors", {method: "POST", body: JSON.stringify({action: "toggle", name: c.name, enabled: on})})
          .catch((e) => notice(e.message));
        if (res?.restart_required) markRestart();
        select("apps");
      }, c.label);
      rows.push(row(c.label, c.description, [testOut, testBtn, sw]));
      if (c.supports_send && c.enabled) {
        rows.push(row("Allow sending email", "Asks for one-time approval before every message with recipients and full text.",
                      toggleSwitch(c.send, async (on) => {
                        const res = await api("/api/connectors", {method: "POST", body: JSON.stringify({action: "send", name: c.name, enabled: on})});
                        if (res?.restart_required) markRestart();
                      }, "Allow sending email")));
      }
    }
    const autoBtn = window.coderunnerShell?.openSystemSettings ? el("button", "linkish small", "Open Automation Settings") : null;
    if (autoBtn) { autoBtn.type = "button"; autoBtn.onclick = () => window.coderunnerShell.openSystemSettings("Privacy_Automation"); }
    body.appendChild(group("Mac apps", rows, "Off until you turn each one on. Writes after reading web pages or email always ask first."));
    if (autoBtn) body.appendChild(autoBtn);
  }

  // ---- 4. Browser ----------------------------------------------------------------------------------
  async function renderBrowser(body) {
    const [data, cfgData, macBrowser] = await Promise.all([
      api("/api/browser"), api("/api/settings"), api("/api/browser/mac").catch(() => ({available: false})),
    ]);
    if (active !== "browser") return;
    body.replaceChildren();
    const modes = [
      ["computer", "Assistant's own computer + Built-in Mac browser",
       "Private Chrome inside the isolated Linux VM you can watch and take over. If a site blocks the VM browser, CodeRunner offers to continue in its own built-in Mac browser tab."],
      ["chrome", "Tab group in your Chrome", "Uses a single CodeRunner tab group in your everyday browser; other tabs stay untouched."],
      ["dedicated", "Separate Chrome window", "Clean Chrome profile on this Mac with no shared logins or cookies."],
    ];
    const modeRows = [];
    for (const [key, label, desc] of modes) {
      const radio = el("input", "set-radio");
      radio.type = "radio"; radio.name = "browser_mode"; radio.checked = data.mode === key;
      const pick = async () => {
        if (data.mode === key) return;
        radio.checked = true;
        await api("/api/browser", {method: "POST", body: JSON.stringify({action: "mode", mode: key})});
        markRestart();
        select("browser");
      };
      radio.onchange = pick;
      const r = row(label, desc, radio);
      r.classList.add("set-choice");
      r.onclick = (ev) => { if (ev.target !== radio) pick(); };
      modeRows.push(r);
    }
    body.appendChild(group("Where the assistant browses", modeRows));

    if (macBrowser && macBrowser.available) {
      const macRows = [];
      const openMacBtn = onOpenMacBrowser ? el("button", "btn", "Open Mac browser") : null;
      if (openMacBtn) {
        openMacBtn.type = "button";
        openMacBtn.onclick = () => { close(); onOpenMacBrowser(); };
      }
      const connected = Boolean(macBrowser.app && macBrowser.app.connected);
      macRows.push(row(
        "CodeRunner's built-in Mac browser",
        connected
          ? "Ready in the workspace panel — its own isolated profile on this Mac, separate from your personal browser. Automatically offered when a site walls off the Linux VM."
          : "Built into the CodeRunner desktop app with its own isolated profile on this Mac.",
        openMacBtn,
      ));
      const importNames = (macBrowser.browsers || []).map((b) => b.name).filter(Boolean);
      if (importNames.length) {
        macRows.push(row(
          "Sign-in cookie import",
          `Can import cookies for a single site on demand from ${importNames.join(" or ")} (via the Mac browser's ⋯ menu).`,
          el("span", "muted small", "On demand"),
        ));
      }
      for (const site of (macBrowser.sites || [])) {
        const rm = el("button", "btn", "Ask next time");
        rm.type = "button";
        rm.onclick = async () => {
          await api(`/api/grants/${site.id}`, {method: "DELETE"}).catch((e) => notice(e.message));
          select("browser");
        };
        macRows.push(row(site.site, "Always allowed to continue in CodeRunner's built-in Mac browser.", rm));
      }
      body.appendChild(group("Built-in Mac browser", macRows));

      // Signed-in session vault (domains & cookie counts only; 1-click per-site sign-out)
      const sessions = macBrowser.sessions || [];
      if (sessions.length) {
        const sesRows = sessions.slice(0, 12).map((s) => {
          const outBtn = el("button", "btn danger", "Sign out");
          outBtn.type = "button";
          outBtn.onclick = async () => {
            outBtn.disabled = true;
            await api("/api/browser/mac/signout", {method: "POST", body: JSON.stringify({domain: s.domain})})
              .catch((e) => notice(e.message));
            select("browser");
          };
          return row(s.domain, `${s.count} session ${s.count === 1 ? "cookie" : "cookies"} stored in Mac browser`, outBtn);
        });
        body.appendChild(group("Signed-in sessions (Mac browser)", sesRows));
      }
    }

    if (data.mode === "chrome") {
      const extDir = (data.extension && data.extension.extension_dir) || "";
      const connBtn = el("button", "btn primary", "Register Chrome bridge");
      connBtn.type = "button";
      connBtn.onclick = async () => {
        connBtn.disabled = true;
        try {
          await api("/api/browser", {method: "POST", body: JSON.stringify({action: "connect_chrome"})});
          markRestart();
          notice("Registered native host. Load the unpacked extension folder in chrome://extensions.");
        } catch (err) { notice(err.message); }
        finally { connBtn.disabled = false; }
      };
      const showBtn = extDir && window.coderunnerShell?.showInFinder ? el("button", "btn", "Show in Finder") : null;
      if (showBtn) { showBtn.type = "button"; showBtn.onclick = () => window.coderunnerShell.showInFinder(extDir); }
      body.appendChild(group("Chrome extension setup", [
        row("Extension folder", extDir || "Generated when registered", [showBtn, connBtn]),
      ], "Open chrome://extensions, turn on Developer mode, and choose Load unpacked with that folder."));
    }

    body.appendChild(group("Site rules", [
      row("Ask before opening any new site", "Prompts for approval before navigating to a website.",
          toggleSwitch(Boolean(cfgGet(cfgData, "policy.ask_new_site", false)), (v) => saveSetting({"policy.ask_new_site": v}), "Ask before new site")),
      row("Never open", (data.block || []).join(", ") || "None", null),
    ]));
  }

  // ---- 5. Routines (Schedules) ---------------------------------------------------------------------
  async function renderRoutines(body) {
    const items = await api("/api/schedules");
    if (active !== "routines") return;
    body.replaceChildren();
    const list = Array.isArray(items) ? items : (items.schedules || []);

    const formRows = [];
    const nameIn = el("input", "set-input"); nameIn.placeholder = "Morning brief";
    const promptIn = el("input", "set-input"); promptIn.placeholder = "Summarise my calendar and reminders for today";
    const freqSel = el("select", "set-select");
    for (const [v, l] of [["weekdays", "Weekdays"], ["daily", "Every day"], ["every 1h", "Every hour"], ["weekly mon", "Mondays"]]) {
      const o = el("option", "", l); o.value = v; freqSel.appendChild(o);
    }
    const timeIn = el("input", "set-input set-time"); timeIn.type = "time"; timeIn.value = "08:00";
    let onlyOnChange = false;
    const changeSw = toggleSwitch(false, (v) => { onlyOnChange = v; }, "Only notify when changed");
    const add = el("button", "btn primary", "Add routine");
    add.type = "button";
    add.onclick = async () => {
      if (!nameIn.value.trim() || !promptIn.value.trim()) { notice("Give the routine a name and a task."); return; }
      const spec = freqSel.value.startsWith("every") ? freqSel.value : `${freqSel.value} ${timeIn.value || "08:00"}`;
      add.disabled = true;
      try {
        await api("/api/schedules", {method: "POST", body: JSON.stringify({
          name: nameIn.value.trim(), spec, prompt: promptIn.value.trim(), on_change: onlyOnChange,
        })});
        select("routines");
      } catch (err) { add.disabled = false; notice(err.message); }
    };
    formRows.push(
      row("Name", "", nameIn),
      row("Task", "", promptIn),
      row("Only notify when changed", "Great for price drops, fee watches, or availability checks.", changeSw),
      row("When", "", [freqSel, timeIn, add]),
    );
    body.appendChild(group("New routine", formRows));

    const existingRows = list.map((it) => {
      const runBtn = el("button", "btn", "Run now");
      runBtn.type = "button";
      runBtn.onclick = async () => {
        await api(`/api/schedules/${it.id}/run`, {method: "POST", body: "{}"}).catch((e) => notice(e.message));
        notice(`Started “${it.name}”.`);
      };
      const delBtn = el("button", "btn danger", "Delete");
      delBtn.type = "button";
      delBtn.onclick = async () => {
        await api(`/api/schedules/${it.id}`, {method: "DELETE"}).catch((e) => notice(e.message));
        select("routines");
      };
      const sw = toggleSwitch(it.enabled !== 0 && it.enabled !== false, (on) =>
        api(`/api/schedules/${it.id}`, {method: "PATCH", body: JSON.stringify({enabled: on})}), it.name);
      const tag = it.on_change ? " · On change only" : "";
      const st = it.last_status ? ` · Last: ${it.last_status}` : "";
      return row(it.name, `${it.spec}${tag}${st} · ${it.prompt}`, [runBtn, sw, delBtn]);
    });
    if (existingRows.length) {
      body.appendChild(group("Your routines", existingRows));
    } else {
      const starters = el("div", "set-chips");
      for (const [n, sp, pr] of [
        ["Morning brief", "weekdays 08:00", "Summarise my calendar and reminders for today."],
        ["Weekly review", "weekly mon 09:00", "List my upcoming meetings and deadlines for the week."],
      ]) {
        const b = el("button", "set-chip", `+ ${n} (${sp})`);
        b.type = "button";
        b.onclick = async () => {
          await api("/api/schedules", {method: "POST", body: JSON.stringify({name: n, spec: sp, prompt: pr})});
          select("routines");
        };
        starters.appendChild(b);
      }
      const empty = el("div", "set-group");
      empty.append(el("div", "set-group-title", "Quick starters"), starters);
      body.appendChild(empty);
    }
  }

  // ---- 6. Privacy & Safety -------------------------------------------------------------------------
  async function renderPrivacy(body) {
    const [cfgData, grants, mems, profData, appCfg] = await Promise.all([
      api("/api/settings"), api("/api/grants"), api("/api/memories"),
      api("/api/profile").catch(() => ({profile: {}, fields: []})),
      window.coderunnerShell?.appSettings?.().catch(() => null) || Promise.resolve(null),
    ]);
    if (active !== "privacy") return;
    body.replaceChildren();
    const s = cfgData.settings || {};

    // Editable About You profile (persistent personal context)
    const prof = (profData && profData.profile) || {};
    const profInputs = {};
    const profFields = [
      ["city", "Home city", "Bengaluru, San Francisco…"],
      ["home_airport", "Home airport", "BLR, SFO, LHR…"],
      ["currency", "Preferred currency", "INR, USD, EUR…"],
      ["dietary", "Dietary preferences", "Vegetarian, vegan, allergies…"],
      ["travel_prefs", "Travel & booking", "Non-stop preferred, economy…"],
      ["notes", "Other lasting notes", "Preferred shops, languages, etc."],
    ];
    const profRows = profFields.map(([k, lbl, ph]) => {
      const inp = el("input", "set-input");
      inp.placeholder = ph;
      inp.value = prof[k] || "";
      profInputs[k] = inp;
      return row(lbl, "", inp);
    });
    const saveProf = el("button", "btn primary", "Save profile");
    saveProf.type = "button";
    saveProf.onclick = async () => {
      saveProf.disabled = true;
      const patch = {};
      for (const [k, inp] of Object.entries(profInputs)) patch[k] = inp.value.trim();
      try {
        await api("/api/profile", {method: "PATCH", body: JSON.stringify(patch)});
        notice("Profile saved locally.");
      } catch (err) { notice(err.message); }
      finally { saveProf.disabled = false; }
    };
    profRows.push(row("Personal context", "Stored only in your local database so you don't repeat defaults.", saveProf));
    body.appendChild(group("About you", profRows));

    // App & Mac behavior
    const appRows = [
      row("Start at login", "Keep the local assistant running quietly in the background.",
          toggleSwitch(Boolean(cfgData.service?.installed), async (on) => {
            await api("/api/service", {method: "POST", body: JSON.stringify({enable: on})}).catch((e) => notice(e.message));
          }, "Start at login")),
      row("Desktop notifications", "Notify you when a background task finishes or needs approval.",
          toggleSwitch(s["owner.desktop_notifications"] !== false,
                       (v) => saveSetting({"owner.desktop_notifications": v}), "Desktop notifications")),
    ];
    if (appCfg && appCfg.settings) {
      const scIn = el("input", "set-input set-short");
      scIn.value = appCfg.settings.quickAskShortcut || "";
      scIn.placeholder = "Alt+Space";
      const scBtn = el("button", "btn", "Save");
      scBtn.type = "button";
      scBtn.onclick = async () => {
        const res = await window.coderunnerShell.appSettings({quickAskShortcut: scIn.value});
        if (!res?.ok) notice(res?.error || "Could not bind shortcut.");
        else notice("Quick Ask shortcut updated.");
      };
      appRows.push(row("Quick Ask shortcut", "Global macOS shortcut to start a task from anywhere.", [scIn, scBtn]));
      appRows.push(row("Developer mode", "Shows Export trace in thread headers.",
                       toggleSwitch(Boolean(appCfg.settings.developerMode),
                                    (v) => window.coderunnerShell.appSettings({developerMode: v}), "Developer mode")));
    }
    const netSel = el("select", "set-select");
    for (const [v, l] of [["default", "Normal internet"], ["proxy", "Approved sites only"], ["none", "Offline (no network)"]]) {
      const o = el("option", "", l); o.value = v; o.selected = (s["sandbox.network"] || "default") === v; netSel.appendChild(o);
    }
    netSel.onchange = () => saveSetting({"sandbox.network": netSel.value});
    appRows.push(row("Sandbox network", "Controls what code running inside the VM can reach.", netSel));

    const vmSel = el("select", "set-select");
    for (const [v, l] of [["auto", "Automatic (on-demand on <=18 GB Macs)"], ["true", "Start only when needed"], ["false", "Always keep running"]]) {
      const o = el("option", "", l); o.value = v; o.selected = String(s["sandbox.on_demand"] ?? "auto") === v; vmSel.appendChild(o);
    }
    vmSel.onchange = () => saveSetting({"sandbox.on_demand": vmSel.value});
    appRows.push(row("Sandbox VM startup", "On-demand startup saves ~2 GB RAM on 8-18 GB Macs.", vmSel));
    body.appendChild(group("This Mac & sandbox", appRows));

    // Standing grants
    const grantList = Array.isArray(grants) ? grants : (grants.grants || []);
    const grantRows = grantList.map((g) => {
      const rev = el("button", "btn danger", "Revoke");
      rev.type = "button";
      rev.onclick = async () => { await api(`/api/grants/${g.id}`, {method: "DELETE"}); select("privacy"); };
      return row(`${g.kind} · ${g.target || "*"}`, `Scope: ${g.scope || "session"}`, rev);
    });
    body.appendChild(group("Standing permissions", grantRows.length ? grantRows : [row("No standing permissions", "Actions ask for approval when needed.", null)]));

    // Memories
    const memList = (mems && mems.memories) || [];
    const memRows = memList.slice(0, 12).map((m) => {
      const fg = el("button", "btn", "Forget");
      fg.type = "button";
      fg.onclick = async () => { await api(`/api/memories/${m.id}`, {method: "DELETE"}); select("privacy"); };
      return row(m.content, m.source === "untrusted" ? "Unverified (from external content)" : "Saved preference", fg);
    });
    body.appendChild(group("What CodeRunner remembers", memRows.length ? memRows : [row("Nothing saved yet", "Preferences you tell the assistant appear here.", null)]));

    // Security self-check
    if ((cfgData.security || []).length) {
      const secRows = cfgData.security.map((c) => row(c.name, c.detail || "", el("span", c.ok ? "set-ok" : "set-warn", c.ok ? "OK" : "Check")));
      body.appendChild(group("Security check", secRows));
    }
  }

  // ---- 7. Voice & Phone ----------------------------------------------------------------------------
  async function renderVoice(body) {
    const [v, ts, devs, ch] = await Promise.all([
      api("/api/voice/settings"), api("/api/tailscale"), api("/api/devices"), api("/api/channels"),
    ]);
    if (active !== "voice") return;
    body.replaceChildren();

    const engSel = el("select", "set-select");
    for (const [val, lbl] of [["auto", "Automatic (Neural Engine first)"], ["parakeet-coreml", "Parakeet Ultra (Neural Engine)"],
                              ["apple", "Apple on-device speech"], ["qwen3-asr", "Qwen3 ASR (30+ languages)"]]) {
      const o = el("option", "", lbl); o.value = val; o.selected = (v.stt || "auto") === val; engSel.appendChild(o);
    }
    engSel.onchange = () => api("/api/voice/settings", {method: "POST", body: JSON.stringify({action: "config", stt: engSel.value})});
    const setupV = el("button", "btn", "Set up / test voice");
    setupV.type = "button";
    setupV.onclick = async () => {
      setupV.disabled = true; setupV.textContent = "Setting up…";
      try {
        await api("/api/voice/settings", {method: "POST", body: JSON.stringify({action: "setup"})});
        notice("Voice setup finished.");
        select("voice");
      } catch (err) { setupV.disabled = false; setupV.textContent = "Set up / test voice"; notice(err.message); }
    };
    body.appendChild(group("On-device dictation", [
      row("Speech engine", "Transcribed locally on this Mac.", engSel),
      row("Voice models", (v.status && v.status.available ? v.status.available.join(", ") : "Ready when set up"), setupV),
    ]));

    // Remote device pairing (1-click inline SVG QR code for mobile PWA over Tailscale)
    const pairCode = el("span", "set-code-inline");
    const qrWrap = el("div", "qr-card");
    qrWrap.hidden = true;
    const pairBtn = el("button", "btn primary", "Show QR code");
    pairBtn.type = "button";
    pairBtn.onclick = async () => {
      pairBtn.disabled = true;
      const res = await api("/api/pair", {method: "POST", body: "{}"}).catch((e) => { notice(e.message); return null; });
      pairBtn.disabled = false;
      if (!res || !res.code) return;
      pairCode.textContent = `Code: ${res.code}`;
      qrWrap.replaceChildren();
      if (res.qr_svg && res.qr_svg.startsWith("<svg")) {
        const img = el("img", "qr-img");
        img.alt = "Pairing QR code";
        img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(res.qr_svg);
        qrWrap.appendChild(img);
      }
      const meta = el("div", "qr-meta");
      meta.append(
        el("div", "set-row-label", `Scan with your phone camera (code ${res.code})`),
        el("div", "muted small", res.tailscale_url
          ? `Opens ${res.tailscale_url} as a mobile web app — tap Share → Add to Home Screen.`
          : "Enable Tailscale Serve for remote access outside Wi-Fi, or open this link on your Mac."),
      );
      const cp = el("button", "btn", "Copy link");
      cp.type = "button";
      cp.onclick = () => { navigator.clipboard?.writeText(res.url || ""); cp.textContent = "Copied"; setTimeout(() => { cp.textContent = "Copy link"; }, 1200); };
      meta.appendChild(cp);
      qrWrap.appendChild(meta);
      qrWrap.hidden = false;
    };
    const devRows = [row("Pair a phone (QR / PWA)", "Scan from iPhone or laptop to monitor tasks, approve actions, and send new work from anywhere.", [pairCode, pairBtn])];
    for (const d of (Array.isArray(devs) ? devs : (devs.devices || []))) {
      const rm = el("button", "btn danger", "Remove");
      rm.type = "button";
      rm.onclick = async () => { await api(`/api/devices/${d.id}`, {method: "DELETE"}); select("voice"); };
      devRows.push(row(d.name || d.id, d.last_seen || "Paired device", rm));
    }
    const devGroup = group("Phone & paired devices", devRows);
    devGroup.appendChild(qrWrap);
    body.appendChild(devGroup);

    // Telegram channel
    const tg = (ch && ch.telegram) || {};
    const tokIn = el("input", "set-input"); tokIn.type = "password";
    tokIn.placeholder = tg.has_token ? "Token saved (paste to replace)" : "BotFather token";
    const tgSave = el("button", "btn", "Save Telegram");
    tgSave.type = "button";
    tgSave.onclick = async () => {
      await api("/api/channels", {method: "POST", body: JSON.stringify({
        channel: "telegram", action: "configure", token: tokIn.value.trim(), enabled: true,
      })}).catch((e) => notice(e.message));
      markRestart();
      notice("Telegram saved. Restart to connect.");
    };
    body.appendChild(group("Messaging channels", [
      row("Telegram bot", tg.enabled ? "Enabled" : "Reach your assistant from Telegram private chats.", [tokIn, tgSave]),
    ]));
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && backdrop) { e.preventDefault(); close(); }
    if ((e.metaKey || e.ctrlKey) && e.key === ",") { e.preventDefault(); open(active); }
  });

  function open(section = "models") {
    if (!backdrop) {
      const built = buildShell();
      titleNode = built.title;
    }
    select(section || "models");
  }

  return {open, close, isOpen: () => Boolean(backdrop)};
}
