"""Benchmark a browser driver (a screenshot-driven model) on recorded states of a booking.com flight search.

  capture STATES_DIR   drive a browser connected through the CodeRunner extension (browser.mode "chrome")
                       through the search, saving at each state a 1440x900 screenshot and the boxes of the
                       element a correct next step would hit, and the three cheapest results at the end
  run BASE_URL MODEL STATES_DIR [OUT.json]
                       ask the model each state once, as a step of the task with the earlier correct steps as
                       history; a click counts when its point falls inside the right element's box (what the
                       broker's hit test would find there); the last state counts when the answer names the
                       three cheapest airlines with their prices

Screenshots are of a third-party site: keep STATES_DIR outside the repository. Example on an M3 Pro (18 GB):
Fara1.5-4B on llama.cpp answered 11 of 11 states, median 7.5 s a step.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import pathlib
import re
import time
import urllib.request

from coderunner.agent import fara

TASK = ("Search booking.com for one-way flights from Bengaluru (BLR) to Delhi (DEL) on Friday 20 November 2026, "
        "1 adult, economy. Do not book or pay for anything. Tell me the three cheapest options with airline, "
        "departure and arrival times, and price.")
VIEW_W, VIEW_H = 1440, 900
RECTS_JS = r"""function() {
  const out = [];
  const add = el => { if (!el || !el.getBoundingClientRect) return; const r = el.getBoundingClientRect();
                      if (r.width > 1 && r.height > 1) out.push([r.left, r.top, r.right, r.bottom]); };
  add(this);
  if (this.labels) for (const l of this.labels) add(l);
  add(this.closest && this.closest('label,a,button,[role=button],[role=option],[role=menuitem],[role=radio],[role=tab],[role=gridcell]'));
  return out;
}"""


# ---- capture ------------------------------------------------------------------------------------------

async def capture(out: pathlib.Path) -> None:
    from coderunner.browser import BrowserBroker, extension
    from coderunner.core import paths
    from coderunner.core.audit import AuditLog
    from coderunner.core.config import Config
    from coderunner.core.store import Store
    from coderunner.safety.policy import Policy

    class Ordinary:
        async def commits(self, state):
            return False

    out.mkdir(parents=True, exist_ok=True)
    config = Config.load()
    audit = AuditLog(paths.audit_path())
    bridge = extension.ChromeBridge(audit=audit)
    await bridge.start()
    b = BrowserBroker(config, Policy(config, Store(":memory:"), audit), audit, decider=Ordinary(),
                      launcher=extension.ExtensionLauncher(bridge, wait=120))
    states: list[dict] = []
    page = {"text": ""}

    def find(pattern: str) -> int | None:
        match = re.search(r"\[(\d+)\] " + pattern, page["text"])
        return int(match.group(1)) if match else None

    async def read() -> str:
        page["text"] = (await b.snapshot(max_chars=400_000)).get("content", "")
        return page["text"]

    async def shoot(name: str, expect: dict) -> None:
        target = b._active
        sid = await b._session(target)
        await b._send(sid, "Emulation.setDeviceMetricsOverride",
                      {"width": VIEW_W, "height": VIEW_H, "deviceScaleFactor": 1, "mobile": False})
        await asyncio.sleep(1.0)
        await read()
        boxes = []
        for pattern in expect.get("click", []):
            ref = find(pattern)
            if ref is not None:
                context = await b._world(sid, target)
                element = b._refs[target][ref]
                rects = await b._call(sid, await b._object(sid, element["backend"], context), RECTS_JS) or []
                boxes += [{"pattern": pattern, "box": box} for box in rects]
        shot = (await b._send(sid, "Page.captureScreenshot", {"format": "png"}))["data"]
        (out / f"{name}.png").write_bytes(base64.b64decode(shot))
        states.append({"name": name, "url": (await b._info(target)).get("url", ""), "expect": expect, "boxes": boxes})
        (out / "states.json").write_text(json.dumps({"task": TASK, "states": states}, indent=1))
        print(f"{name}: {len(boxes)} target boxes")

    async def click(pattern: str) -> None:
        ref = find(pattern)
        if ref is None or not (await b.click(ref)).get("ok"):
            raise SystemExit(f"could not click {pattern}")

    day = r'(?:button|gridcell|checkbox|link) "[^"]*(?:\b20\b[^"]*Nov|Nov[^"]*\b20\b)[^"]*"'
    try:
        await b.open("https://www.booking.com/")
        await asyncio.sleep(3)
        await read()
        if find(r'button "Dismiss sign-in info') is not None:
            await shoot("00_signin_popup", {"click": [r'button "Dismiss sign-in info']})
            await click(r'button "Dismiss sign-in info')
        steps = [("01_home", r'(?:link|menuitem) "Flights"', 3), ("02_form", r'radio "One-way"', 0),
                 ("03_one_way", r'button "Going to"', 0)]
        for name, pattern, pause in steps:
            await shoot(name, {"click": [pattern]})
            await click(pattern)
            await asyncio.sleep(pause)
        await shoot("04_going_to_open", {"click": [r'combobox "Arrival airport'], "type": "Delhi"})
        await b.type(find(r'combobox "Arrival airport'), "Delhi")
        await asyncio.sleep(2)
        for name, pattern in [("05_suggestions", r'option "DEL Delhi'), ("06_destination_set", r'button "Travel date'),
                              ("07_date_picker", day)]:
            await shoot(name, {"click": [pattern]})
            await click(pattern)
            await asyncio.sleep(1)
        await read()
        if find(r'button "Done"') is not None:
            await shoot("07b_date_done", {"click": [r'button "Done"']})
            await click(r'button "Done"')
        await shoot("08_form_complete", {"click": [r'button "(?:Explore|Search)"']})
        await click(r'button "(?:Explore|Search)"')
        await asyncio.sleep(6)
        await shoot("09_results_best", {"click": [r'button "Sort by[^"]*"', r'tab "Cheapest[^"]*"']})
        await click(r'button "Sort by[^"]*"')
        await asyncio.sleep(1)
        await shoot("10_sort_menu", {"click": [r'menuitem "Cheapest"']})
        await click(r'menuitem "Cheapest"')
        await asyncio.sleep(5)
        text = await read()
        truth = []
        for body in re.findall(r"Flight option \d+ of \d+\n(.*?)(?=Flight option \d+ of|\Z)", text, re.S)[:3]:
            lines = [line for line in body.splitlines() if not line.startswith("[")]
            truth.append({"airline": next((x for x in lines if re.fullmatch(r"[A-Z][A-Za-z ]+", x)
                                           and x not in ("Direct", "Saver")), ""),
                          "price": next((x for x in lines if x.startswith("INR")), "")})
        await shoot("11_results_cheapest", {"answer": truth})
    finally:
        await b.stop()
        await bridge.stop()


# ---- run ----------------------------------------------------------------------------------------------

def run(base_url: str, model: str, states_dir: pathlib.Path, out: pathlib.Path | None) -> None:
    data = json.loads((states_dir / "states.json").read_text())
    states = data["states"]

    def correct_step(state: dict) -> str:
        if state["expect"].get("type"):
            args = {"action": "type", "text": state["expect"]["type"]}
        else:
            x0, y0, x1, y1 = state["boxes"][0]["box"]
            args = {"action": "left_click", "coordinate": [round((x0 + x1) / 2 * fara.DISPLAY / VIEW_W),
                                                            round((y0 + y1) / 2 * fara.DISPLAY / VIEW_H)]}
        call = json.dumps({"name": "computer_use", "arguments": args})
        return f"Next step for: {state['name'][3:].replace('_', ' ')}.\n<tool_call>\n{call}\n</tool_call>"

    def messages(i: int) -> list:
        msgs = [{"role": "system", "content": fara.system_prompt()}]
        for j in range(i):
            msgs += [{"role": "user", "content": data["task"] if j == 0 else
                      f"Current URL: {states[j]['url'][:100]}\n{fara.NEXT}"},
                     {"role": "assistant", "content": correct_step(states[j])}]
        image = base64.b64encode((states_dir / f"{states[i]['name']}.png").read_bytes()).decode()
        text = data["task"] if i == 0 else f"Current URL: {states[i]['url'][:100]}\n{fara.NEXT}"
        return msgs + [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}, {"type": "text", "text": text}]}]

    def judge(state: dict, args: dict | None) -> tuple[bool, str]:
        if not args:
            return False, "no action"
        expect, action = state["expect"], args.get("action")
        if "answer" in expect:
            answer = json.dumps(args, ensure_ascii=False)
            digits = re.sub(r"\D", "", answer)
            found = sum(1 for t in expect["answer"]
                        if t["airline"].split()[0] in answer and re.sub(r"\D", "", t["price"])[:4] in digits)
            return found == len(expect["answer"]), f"{action}: {found}/{len(expect['answer'])} named with price"
        if action == "type" and expect.get("type", "").lower() in str(args.get("text", "")).lower() and expect.get("type"):
            return True, f"type {args.get('text')!r}"
        point = fara.to_pixels(args.get("coordinate"), VIEW_W, VIEW_H)
        if action in ("left_click", "double_click", "triple_click") and point:
            x, y = point
            hit = any(b["box"][0] - 4 <= x <= b["box"][2] + 4 and b["box"][1] - 4 <= y <= b["box"][3] + 4
                      for b in state["boxes"])
            return hit, f"{action} at ({x:.0f}, {y:.0f})"
        return False, str(action)

    results = []
    for i, state in enumerate(states):
        body = {"model": model, "messages": messages(i), "temperature": 0, "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False}}
        began = time.monotonic()
        request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(request, timeout=900))
        seconds = time.monotonic() - began
        ok, detail = judge(state, fara.parse(reply["choices"][0]["message"].get("content") or "")[1])
        results.append({"state": state["name"], "ok": ok, "detail": detail, "seconds": round(seconds, 1)})
        print(f"{state['name']:22s} {'OK  ' if ok else 'MISS'} {detail:45s} {seconds:5.1f} s", flush=True)
    times = sorted(r["seconds"] for r in results)
    print(f"{sum(r['ok'] for r in results)}/{len(results)} right, median step {times[len(times) // 2]} s")
    if out:
        out.write_text(json.dumps({"model": model, "results": results}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capture").add_argument("states_dir", type=pathlib.Path)
    r = sub.add_parser("run")
    r.add_argument("base_url"), r.add_argument("model"), r.add_argument("states_dir", type=pathlib.Path)
    r.add_argument("out", type=pathlib.Path, nargs="?")
    args = parser.parse_args()
    if args.command == "capture":
        asyncio.run(capture(args.states_dir))
    else:
        run(args.base_url, args.model, args.states_dir, args.out)


if __name__ == "__main__":
    main()
