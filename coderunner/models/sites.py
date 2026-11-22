"""Local stand-ins for the sites and files owners ask about, used by the task suite (tasks.py).

They have the same hurdles as the real ones: a flight search that starts on round trip, airport fields that
only accept a pick from their suggestions, a calendar to page through, results sorted by "best" rather than
price and wrapped in filters and banners; a library website with the facts on different pages; a folder of
documents where the answer needs two of them. Flights and prices come from the route and date, so the checker
knows the answer and no model can have learned it.
"""

from __future__ import annotations

import datetime
import html
import json
import pathlib
import random
import re
import socket
from typing import Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route

AIRPORTS = [  # code, city, airport, other names
    ("BLR", "Bengaluru", "Kempegowda International", ["Bangalore"]),
    ("DEL", "Delhi", "Indira Gandhi International", ["New Delhi"]),
    ("HDO", "Delhi", "Hindon (Ghaziabad)", ["Ghaziabad"]),
    ("BOM", "Mumbai", "Chhatrapati Shivaji Maharaj International", ["Bombay"]),
    ("NMI", "Mumbai", "Navi Mumbai International", ["Navi Mumbai"]),
    ("MAA", "Chennai", "Chennai International", ["Madras"]),
    ("CCU", "Kolkata", "Netaji Subhas Chandra Bose International", ["Calcutta"]),
    ("HYD", "Hyderabad", "Rajiv Gandhi International", []),
    ("PNQ", "Pune", "Pune International", []),
    ("GOI", "Goa", "Dabolim", ["Goa Dabolim"]),
    ("GOX", "Goa", "Manohar International (Mopa)", ["Mopa"]),
    ("JAI", "Jaipur", "Jaipur International", []),
    ("COK", "Kochi", "Cochin International", ["Cochin"]),
    ("AMD", "Ahmedabad", "Sardar Vallabhbhai Patel International", []),
    ("LKO", "Lucknow", "Chaudhary Charan Singh International", []),
    ("DXB", "Dubai", "Dubai International", []),
    ("SIN", "Singapore", "Changi", []),
    ("LHR", "London", "Heathrow", []),
]
CITY = {code: city for code, city, _, _ in AIRPORTS}
AIRLINES = (("IndiGo", "6E"), ("Air India", "AI"), ("Akasa Air", "QP"), ("SpiceJet", "SG"), ("Air India Express", "IX"))
CABINS = {"economy": ("Economy", 1.0), "premium": ("Premium economy", 1.8), "business": ("Business", 3.2)}
SORTS = {"best": "Best", "cheapest": "Cheapest", "fastest": "Fastest"}
PAGE_RESULTS = 15            # like the real sites, the rest behind "show more"


def flights(origin: str, dest: str, day: str) -> list[dict]:
    """One-way economy fares for one adult, in the site's "best" order. The three cheapest are never among
    the first four, so reading the top of the default list gives the wrong answer."""
    rnd = random.Random(f"{origin}-{dest}-{day}")
    out, used = [], set()
    for rank in range(18):
        airline, code = rnd.choice(AIRLINES)
        stops = 1 if rnd.random() < 0.3 else 0
        price = rnd.randrange(6800, 15500)
        while price in used:
            price += 7
        used.add(price)
        out.append({"rank": rank, "airline": airline, "number": f"{code} {rnd.randint(200, 2999)}",
                    "depart": rnd.randrange(5 * 60, 22 * 60, 5),
                    "minutes": rnd.randrange(290, 600, 5) if stops else rnd.randrange(160, 190, 5),
                    "stops": stops, "price": price})
    cheap = rnd.sample(out[4:], 3)
    for f, price in zip(cheap, sorted(rnd.sample(range(5100, 6700), 3))):
        f["price"] = price
        if not f["stops"] and f["depart"] > 18 * 60:      # so the evening question has its own answer
            f["depart"] = rnd.randrange(6 * 60, 17 * 60, 5)
    for f in rnd.sample([f for f in out if f not in cheap], 2):      # at least two direct evening flights
        f.update(stops=0, depart=rnd.randrange(18 * 60 + 5, 21 * 60 + 35, 5), minutes=rnd.randrange(160, 190, 5))
    return out


def cheapest(origin: str, dest: str, day: str, n: int = 3) -> list[dict]:
    return sorted(flights(origin, dest, day), key=lambda f: f["price"])[:n]


def evening(origin: str, dest: str, day: str) -> dict:
    """The cheapest direct flight leaving after 18:00."""
    return min((f for f in flights(origin, dest, day) if not f["stops"] and f["depart"] > 18 * 60), key=lambda f: f["price"])


def clock(minutes: int) -> str:
    day, minutes = divmod(minutes, 24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}" + (f" +{day}" if day else "")


def rupees(amount: int) -> str:
    """Indian grouping, as the real sites show it: 1,23,456."""
    text = str(amount)
    if len(text) > 3:
        head, tail = text[:-3], text[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        text = ",".join(([head] if head else []) + groups + [tail])
    return f"₹{text}"


def _page(title: str, body: str, style: str = "") -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;color:#1a1a1a}}header,footer{{background:#0b3d91;color:#fff;
padding:10px 24px}}header a,footer a{{color:#fff;margin-right:14px}}main{{padding:16px 24px;max-width:1100px}}
[hidden]{{display:none!important}}{style}</style></head><body>{body}</body></html>""")


# ---- SkyFare: flight search -------------------------------------------------------------------------------

FLIGHT_STYLE = """.row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin:10px 0}label{display:block}
.combo{position:relative}[role=listbox]{position:absolute;z-index:5;background:#fff;border:1px solid #888;list-style:none;
margin:0;padding:0;width:360px}[role=option]{padding:6px 8px;cursor:pointer}[role=option][aria-selected=true]{background:#dde8ff}
#calendar{position:absolute;top:calc(100% + 4px);left:0;z-index:6;background:#fff;border:1px solid #888;padding:10px;width:320px}
#calendar button.day{width:40px;height:32px}.week{display:flex;gap:4px;margin:4px 0}.week span{width:40px}
#alert{color:#b00020;font-weight:600}#cookies{position:fixed;bottom:0;left:0;right:0;background:#222;color:#fff;padding:12px 24px}
.layout{display:flex;gap:24px}aside{width:230px}article{border:1px solid #ccc;border-radius:8px;padding:10px 14px;margin:8px 0}
.price{font-size:20px;font-weight:700}nav.sort a{margin-right:16px}nav.sort a[aria-current]{font-weight:700}"""

FLIGHT_HEADER = """<header><a href="/flights/"><b>SkyFare</b></a><a href="/flights/">Flights</a><a href="/flights/hotels">Stays</a>
<a href="/flights/cars">Car rental</a><a href="/flights/deals">Deals</a><a href="/flights/help">Help</a>
<a href="/flights/signin">Sign in</a><a href="/flights/signin">Register</a></header>"""
FLIGHT_FOOTER = """<footer><a href="/flights/about">About SkyFare</a><a href="/flights/help">Customer service</a>
<a href="/flights/terms">Terms</a><a href="/flights/privacy">Privacy</a><a href="/flights/cookies">Cookie settings</a></footer>"""

FLIGHT_HOME = """<main><h1>Find cheap flights</h1>
<form id="search" action="/flights/results" method="get" autocomplete="off">
<div class="row"><label><input type="radio" name="trip" value="round" checked> Round trip</label>
<label><input type="radio" name="trip" value="oneway"> One way</label>
<label><input type="radio" name="trip" value="multi" disabled> Multi-city (coming soon)</label></div>
<div class="row">
<div class="combo"><label for="from-input">Leaving from</label>
<input id="from-input" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="from-list" placeholder="City or airport">
<ul id="from-list" role="listbox" aria-label="Leaving from suggestions" hidden></ul><input type="hidden" name="from"></div>
<div class="combo"><label for="to-input">Going to</label>
<input id="to-input" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="to-list" placeholder="City or airport">
<ul id="to-list" role="listbox" aria-label="Going to suggestions" hidden></ul><input type="hidden" name="to"></div>
<div class="combo"><button type="button" id="depart-btn" aria-haspopup="dialog">Departure date</button>
<input type="hidden" name="depart">
<div id="calendar" role="dialog" aria-label="Choose a date" hidden><div class="row">
<button type="button" id="prev" aria-label="Previous month">&lsaquo;</button>
<button type="button" id="next" aria-label="Next month">&rsaquo;</button></div><div id="grid"></div><button type="button" id="close-cal">Close</button></div></div>
<div class="combo" id="return-box"><button type="button" id="return-btn" aria-haspopup="dialog">Return date</button>
<input type="hidden" name="return"></div>
<label>Travellers <select name="adults">ADULTS</select></label>
<label>Cabin <select name="cabin"><option value="economy">Economy</option><option value="premium">Premium economy</option>
<option value="business">Business</option></select></label>
<button type="submit">Search</button></div>
<p id="alert" role="alert" hidden></p>
<label><input type="checkbox" name="direct"> Direct flights only</label>
</form>
<h2>Popular right now</h2><p><a href="/flights/deals">Goa from ₹3,899</a> · <a href="/flights/deals">Dubai from ₹11,250</a> ·
<a href="/flights/deals">Singapore from ₹13,990</a> · <a href="/flights/deals">Kochi from ₹4,120</a></p>
<h2>Why book with SkyFare</h2><p>No hidden fees. Free cancellation on selected fares. 24/7 customer service.</p>
</main>
<div id="cookies" role="region" aria-label="Cookie consent">We use cookies to improve your experience.
<button type="button" onclick="this.parentNode.hidden=true">Accept</button>
<button type="button" onclick="this.parentNode.hidden=true">Reject</button></div>
<script>
const AIRPORTS = DATA, TODAY = "TODAYISO";
const pad = n => String(n).padStart(2, "0");
const iso = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const form = document.getElementById("search"), alertBox = document.getElementById("alert");
function combo(field) {
  const input = document.getElementById(field + "-input"), list = document.getElementById(field + "-list"),
        hidden = form.elements[field];
  let found = [], active = 0;
  const mark = () => [...list.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); };
  function choose(i) { const a = found[i]; input.value = `${a[1]} (${a[0]})`; hidden.value = a[0]; close(); }
  input.addEventListener("input", () => {
    hidden.value = "";
    const q = input.value.trim().toLowerCase();
    const score = a => (a[0].toLowerCase() === q ? 3 : 0) + (a[1].toLowerCase().startsWith(q) ? 2 : 0);
    found = q ? AIRPORTS.filter(a => [a[0], a[1], a[2], ...a[3]].some(s => s.toLowerCase().includes(q))) : [];
    found = found.sort((a, b) => score(b) - score(a)).slice(0, 6);
    active = 0;
    list.innerHTML = found.map((a, i) => `<li role="option" id="${field}-opt-${i}">${a[1]} (${a[0]}) · ${a[2]}</li>`).join("");
    mark();
    list.hidden = !found.length;
    input.setAttribute("aria-expanded", String(!!found.length));
  });
  list.addEventListener("mousedown", e => e.preventDefault());
  list.addEventListener("click", e => { const li = e.target.closest("li"); if (li) choose(+li.id.split("-").pop()); });
  input.addEventListener("keydown", e => {
    if (list.hidden) return;
    if (e.key === "ArrowDown") { active = Math.min(active + 1, found.length - 1); mark(); e.preventDefault(); }
    if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); mark(); e.preventDefault(); }
    if (e.key === "Enter") { choose(active); e.preventDefault(); }
    if (e.key === "Escape") close();
  });
}
combo("from"); combo("to");
const today = new Date(TODAY + "T00:00:00"), cal = document.getElementById("calendar");
let shown = new Date(today.getFullYear(), today.getMonth(), 1), picking = "depart";
function month(start) {
  const first = (start.getDay() + 6) % 7, days = new Date(start.getFullYear(), start.getMonth() + 1, 0).getDate();
  let rows = `<h3>${start.toLocaleString("en-GB", {month: "long", year: "numeric"})}</h3>` +
    '<div class="week" aria-hidden="true"><span>Mo</span><span>Tu</span><span>We</span><span>Th</span><span>Fr</span>' +
    '<span>Sa</span><span>Su</span></div><div class="week">' + "<span></span>".repeat(first);
  for (let d = 1; d <= days; d++) {
    const day = new Date(start.getFullYear(), start.getMonth(), d);
    const label = day.toLocaleDateString("en-GB", {weekday: "long", day: "numeric", month: "long", year: "numeric"});
    rows += `<button type="button" class="day" data-date="${iso(day)}" aria-label="${label}, choose as ${picking} date"` +
      `${day < today ? " disabled" : ""}>${d}</button>`;
    if ((first + d) % 7 === 0) rows += '</div><div class="week">';
  }
  return rows + "</div>";
}
function draw() {   // two months side by side, as the real sites show them
  document.getElementById("grid").innerHTML = month(shown) + month(new Date(shown.getFullYear(), shown.getMonth() + 1, 1));
}
function openCal(which) { picking = which; draw(); cal.hidden = false; }
document.getElementById("depart-btn").onclick = () => openCal("depart");
document.getElementById("return-btn").onclick = () => openCal("return");
document.getElementById("prev").onclick = () => { shown = new Date(shown.getFullYear(), shown.getMonth() - 1, 1); draw(); };
document.getElementById("next").onclick = () => { shown = new Date(shown.getFullYear(), shown.getMonth() + 1, 1); draw(); };
document.getElementById("close-cal").onclick = () => { cal.hidden = true; };
document.getElementById("grid").addEventListener("click", e => {
  const b = e.target.closest("button.day"); if (!b || b.disabled) return;
  form.elements[picking].value = b.dataset.date;
  const d = new Date(b.dataset.date + "T00:00:00");
  document.getElementById(picking + "-btn").textContent = (picking === "depart" ? "Departure: " : "Return: ") +
    d.toLocaleDateString("en-GB", {weekday: "short", day: "numeric", month: "short", year: "numeric"});
  cal.hidden = true;
});
form.addEventListener("change", () => {
  document.getElementById("return-box").hidden = form.elements.trip.value !== "round";
});
form.addEventListener("submit", e => {
  const problems = [];
  if (!form.elements.from.value) problems.push("Choose where you are leaving from in the list of suggestions.");
  if (!form.elements.to.value) problems.push("Choose where you are going in the list of suggestions.");
  if (!form.elements.depart.value) problems.push("Choose a departure date.");
  if (form.elements.trip.value === "round" && !form.elements.return.value)
    problems.push("Choose a return date, or switch to One way.");
  if (problems.length) { e.preventDefault(); alertBox.textContent = problems.join(" "); alertBox.hidden = false; }
});
</script>"""


def _fare(f: dict, query: dict) -> int:
    """What the results page charges: the one-way economy fare, scaled for travellers, cabin and a return leg."""
    price = f["price"]
    if query["trip"] == "round":
        price += random.Random(f"{f['number']}-{query['return']}").randrange(5200, 9800)
    return round(price * CABINS[query["cabin"]][1]) * query["adults"]


class Sites:
    """The test sites, served on 127.0.0.1. `visits` records which pages were opened, for the checks."""

    def __init__(self, today: datetime.date | None = None):
        self.today = today or datetime.date.today()
        self.visits: list[str] = []
        self.host = "127.0.0.1"         # the address they are served on (the sandbox's gateway, for its browser)
        self.port = 0
        self.app = Starlette(routes=[
            Route("/", lambda r: RedirectResponse("/flights/")),
            Route("/flights/", self.flight_home), Route("/flights/results", self.flight_results),
            Route("/flights/checkout", self.flight_checkout), Route("/flights/payment", self.flight_payment),
            Route("/flights/{other}", self.flight_other),
            Route("/library/", self.library), Route("/library/{page}", self.library),
            Route("/kelvin/", self.kelvin), Route("/kelvin/search", self.kelvin_search),
            Route("/kelvin/guide/{model}", self.kelvin_guide), Route("/kelvin/{page}", self.kelvin),
        ])

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Serve in this event loop; returns the server task (cancel it to stop)."""
        import asyncio

        import uvicorn

        with socket.socket() as sock:
            sock.bind((self.host, 0))
            self.port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning",
                                               access_log=False, lifespan="off"))
        server.capture_signals = lambda: __import__("contextlib").nullcontext()
        task = asyncio.create_task(server.serve())
        while not server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.05)
        return server, task

    def _visit(self, request: Request) -> None:
        self.visits.append(request.url.path + (f"?{request.url.query}" if request.url.query else ""))

    # ---- flights

    async def flight_home(self, request: Request):
        self._visit(request)
        page = (FLIGHT_HOME.replace("DATA", json.dumps(AIRPORTS)).replace("TODAYISO", self.today.isoformat())
                .replace("ADULTS", "".join(f'<option value="{n}">{n} adult{"s" if n > 1 else ""}</option>'
                                           for n in range(1, 10))))
        return _page("SkyFare – cheap flights, hotels and car rental", FLIGHT_HEADER + page + FLIGHT_FOOTER, FLIGHT_STYLE)

    async def flight_other(self, request: Request):
        self._visit(request)
        name = request.path_params["other"].replace("-", " ").capitalize()
        return _page(f"SkyFare – {name}", FLIGHT_HEADER + f"<main><h1>{html.escape(name)}</h1><p>This page is not "
                     "available right now.</p><p><a href='/flights/'>Back to flight search</a></p></main>" + FLIGHT_FOOTER,
                     FLIGHT_STYLE)

    async def flight_checkout(self, request: Request):
        self._visit(request)
        return _page("SkyFare – Who's flying?", FLIGHT_HEADER + """<main><h1>Who's flying?</h1><form>
<label>First name <input name="first"></label><label>Last name <input name="last"></label>
<label>Email <input name="email"></label></form><a href="/flights/payment">Continue to payment</a></main>""",
                     FLIGHT_STYLE)

    async def flight_payment(self, request: Request):
        self._visit(request)
        return _page("SkyFare – Payment", FLIGHT_HEADER + """<main><h1>Pay for your flight</h1><form>
<label>Card number <input autocomplete="cc-number"></label><button type="button">Pay now</button></form></main>""",
                     FLIGHT_STYLE)

    async def flight_results(self, request: Request):
        self._visit(request)
        p = request.query_params
        try:
            query = {"from": p["from"].upper(), "to": p["to"].upper(), "depart": datetime.date.fromisoformat(p["depart"]),
                     "trip": p.get("trip", "round"), "return": p.get("return", ""), "adults": int(p.get("adults", 1)),
                     "cabin": p.get("cabin", "economy")}
            if query["from"] not in CITY or query["to"] not in CITY or query["cabin"] not in CABINS \
                    or not 1 <= query["adults"] <= 9 or (query["trip"] == "round" and not query["return"]):
                raise ValueError
        except (KeyError, ValueError):
            return _page("SkyFare – search", FLIGHT_HEADER + "<main><h1>We couldn't run that search</h1><p>Some search "
                         "details are missing or not valid. <a href='/flights/'>Search again</a></p></main>", FLIGHT_STYLE)
        sort = p.get("sort", "best") if p.get("sort") in SORTS else "best"
        found = flights(query["from"], query["to"], query["depart"].isoformat())
        stops, airlines = p.get("stops", "any"), p.getlist("airline")
        if p.get("direct"):
            stops = "0"
        shown = [f for f in found if (stops == "any" or f["stops"] <= int(stops)) and (not airlines or f["airline"] in airlines)]
        shown.sort(key={"best": lambda f: f["rank"], "cheapest": lambda f: (_fare(f, query), f["rank"]),
                        "fastest": lambda f: (f["minutes"], f["rank"])}[sort])
        keep = [(k, v) for k, v in p.multi_items() if k not in ("sort", "more")]
        from urllib.parse import urlencode

        def link(**changes) -> str:
            items = [(k, v) for k, v in keep if k not in changes] + list(changes.items())
            return "/flights/results?" + html.escape(urlencode(items))

        def count(rows) -> str:
            rows = list(rows)
            return f"From {rupees(min(_fare(f, query) for f in rows))} · {len(rows)}" if rows else "0"

        travellers = f"{query['adults']} adult{'s' if query['adults'] > 1 else ''}"
        day = query["depart"].strftime("%a, %-d %b")
        note = ("Prices are round trip " if query["trip"] == "round" else "Prices are one way ") + \
            (f"for all {travellers}" if query["adults"] > 1 else "per person") + ", including taxes and fees."
        change = "onclick=\"location.href='/flights/'\""
        search = (f"<form aria-label='Search for flights'><p>Trip type</p>"
                  f"<label><input type=radio name=t {change}{' checked' if query['trip'] == 'round' else ''}> Round trip</label>"
                  f"<label><input type=radio name=t {change}{' checked' if query['trip'] == 'oneway' else ''}> One way</label>"
                  f"<label>Cabin <select {change}><option>{CABINS[query['cabin']][0]}</option></select></label>"
                  f"<button type=button {change}>Leaving from {query['from']} {CITY[query['from']]}</button>"
                  f"<button type=button {change} aria-label='Swap origin and destination'>⇄</button>"
                  f"<button type=button {change}>Going to {query['to']} {CITY[query['to']]}</button>"
                  f"<button type=button {change}>Travel date {day}</button>"
                  f"<button type=button {change}>Travellers {travellers}</button>"
                  f"<button type=button {change}>Search</button></form>")
        hidden = "".join(f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">' for k, v in keep
                         if k not in ("stops", "airline", "direct"))
        hours = (("00:00–05:59", 0, 6), ("06:00–11:59", 6, 12), ("12:00–17:59", 12, 18), ("18:00–23:59", 18, 24))
        def slots(title: str, minute: Callable[[dict], int]) -> str:
            return f"<p>{title}</p>" + "".join(
                f"<label><input type='checkbox' disabled> {label} ({sum(lo * 60 <= minute(f) % 1440 < hi * 60 for f in found)})</label>"
                for label, lo, hi in hours)
        times = slots(f"Departs {CITY[query['from']]}", lambda f: f["depart"]) + \
            slots(f"Arrives in {CITY[query['to']]}", lambda f: f["depart"] + f["minutes"])
        filters = (f'<aside><h2>Filters</h2><p>Showing {len(shown)} results</p><form method="get" action="/flights/results">'
                   f'{hidden}<input type="hidden" name="sort" value="{sort}"><h3>Stops</h3>' + "".join(
                       f'<label><input type="radio" name="stops" value="{v}"{" checked" if v == stops else ""}> {t} '
                       f'{count(f for f in found if v == "any" or f["stops"] <= int(v))}</label>'
                       for v, t in (("any", "Any"), ("0", "Direct only"), ("1", "1 stop at most"))) +
                   "<h3>Baggage</h3><label><input type='checkbox' disabled> Checked bag included (shows fares with a checked "
                   "bag; see fare options for all results)</label><h3>Airlines</h3>" + "".join(
                       f'<label><input type="checkbox" name="airline" value="{a}"{" checked" if a in airlines else ""}> {a} '
                       f'({sum(f["airline"] == a for f in found)})</label>' for a, _ in AIRLINES) +
                   f"<h3>Flight times</h3>{times}<h3>Duration</h3><label>Maximum travel time "
                   "<input type='range' min='2' max='12' value='12' disabled> 12 hours</label>"
                   "<button type='submit'>Apply filters</button></form></aside>")
        page = shown if p.get("more") else shown[:PAGE_RESULTS]
        rows = []
        for i, f in enumerate(page, 1):
            arrive = f["depart"] + f["minutes"]
            fare = random.Random(f["number"]).choice(("Saver", "Saver", "Flexi", "Value"))
            rows.append(f"<article><p>Flight option {i} of {len(shown)}</p>{'<p>Best</p>' if i == 1 and sort == 'best' else ''}"
                        f"<p><b>{clock(f['depart'])}</b> {query['from']} · {query['depart']:%-d %b}</p>"
                        f"<p>{'Direct' if not f['stops'] else '1 stop · Hyderabad'}</p><p>{f['minutes'] // 60}h {f['minutes'] % 60:02d}m</p>"
                        f"<p><b>{clock(arrive)}</b> {query['to']}</p><p>{f['airline']} · {f['number']}</p><p>{fare}</p>"
                        f"<button type=button>Personal item included, cabin bag included{', checked bag included' if fare != 'Saver' else ''}</button>"
                        f"<p class='price'>{rupees(_fare(f, query))}</p>"
                        f"<button type=button aria-expanded=false>Open price breakdown, flight option {i}</button>"
                        f"<button type=button aria-expanded=false>Fare options</button>"
                        f"<a href='/flights/checkout?flight={f['number'].replace(' ', '')}'>Select</a></article>")
        more = (f"<a href='{link(sort=sort, more=1)}'>Show {len(shown) - len(page)} more flights</a>"
                if len(page) < len(shown) else "")
        tabs = " ".join(f'<a href="{link(sort=key)}"{" aria-current=page" if key == sort else ""}>{label}</a>'
                        for key, label in SORTS.items())
        body = (f"{FLIGHT_HEADER}<main>{search}<p>{CITY[query['from']]} ({query['from']}) → {CITY[query['to']]} "
                f"({query['to']}) · {day} · {travellers} · {CABINS[query['cabin']][0]}</p>"
                f"<div class='layout'>{filters}<section><h1>We found {len(shown)} flights</h1><nav class='sort' "
                f"aria-label='Sort by'>Sort by: {tabs}</nav><p>{note}</p><p><a href='/flights/signin'>Sign in</a> to get "
                f"price alerts and save 10% with member deals.</p>{''.join(rows) or '<p>No flights match these filters.</p>'}"
                f"{more}</section></div></main>{FLIGHT_FOOTER}")
        return _page(f"SkyFare – {CITY[query['from']]} to {CITY[query['to']]} flights", body, FLIGHT_STYLE)

    # ---- the library

    async def library(self, request: Request):
        self._visit(request)
        name = request.path_params.get("page", "")
        title, content = LIBRARY.get(name, ("Page not found", "<p>We couldn't find that page.</p>"))
        nav = "".join(f'<a href="/library/{key}">{label}</a>' for key, label in LIBRARY_NAV)
        return _page(f"{title} – Riverside Public Library",
                     f"<header><b>Riverside Public Library</b> {nav}</header><main><h1>{title}</h1>{content}</main>"
                     "<footer><a href='/library/contact'>Contact</a><a href='/library/support'>Support us</a>"
                     "<a href='/library/accessibility'>Accessibility</a> © Riverside Public Library Trust</footer>")


    # ---- the appliance maker's support site

    def _kelvin(self, title: str, content: str) -> HTMLResponse:
        nav = "".join(f'<a href="/kelvin/{key}">{label}</a>' for key, label in KELVIN_NAV)
        return _page(f"{title} – Kelvin Appliances", f"<header><b>Kelvin Appliances</b> {nav}</header><main>"
                     f"<form action='/kelvin/search'><label>Search support by model number or product "
                     f"<input name='q' placeholder='e.g. DW-412'></label><button>Search</button></form>"
                     f"<h1>{title}</h1>{content}</main><footer><a href='/kelvin/contact'>Contact</a>"
                     "<a href='/kelvin/register'>Register a product</a><a href='/kelvin/privacy'>Privacy</a></footer>")

    async def kelvin(self, request: Request):
        self._visit(request)
        page = request.path_params.get("page", "")
        if page == "products":
            return self._kelvin("Products", "".join(
                f"<p><a href='/kelvin/guide/{model}'>{name} ({model})</a></p>" for model, (name, *_) in KELVIN_GUIDES.items()))
        if page:
            return self._kelvin(page.replace("-", " ").capitalize(), "<p>This page is being updated.</p>")
        return self._kelvin("Support", "<p>Find manuals, care guides and service for your Kelvin appliance.</p>"
                            "<h2>Popular topics</h2><p><a href='/kelvin/service'>Book a repair visit</a> · "
                            "<a href='/kelvin/register'>Register your product</a> · <a href='/kelvin/products'>All "
                            "products and care guides</a> · <a href='/kelvin/warranty'>Warranty terms</a></p>"
                            "<p>Kelvin Care: 1800 555 0199 (toll free, 8 am to 8 pm, all days)</p>")

    async def kelvin_search(self, request: Request):
        self._visit(request)
        q = request.query_params.get("q", "")
        key = re.sub(r"[\s-]", "", q).lower()
        found = [(m, g) for m, g in KELVIN_GUIDES.items()
                 if key and (key in m.replace("-", "").lower() or key in re.sub(r"[\s-]", "", g[0]).lower())]
        rows = "".join(f"<article><h3><a href='/kelvin/guide/{m}'>{m} · {g[0]}: care and maintenance guide</a></h3>"
                       f"<p><a href='/kelvin/manuals'>{m} user manual (PDF, 4.2 MB)</a></p></article>" for m, g in found)
        return self._kelvin(f"Results for “{html.escape(q)}”", rows or "<p>Nothing matches. Check the model number on "
                            "the label inside the door or on your invoice.</p>")

    async def kelvin_guide(self, request: Request):
        self._visit(request)
        model = request.path_params["model"].upper()
        if model not in KELVIN_GUIDES:
            return self._kelvin("Guide not found", "<p>We have no care guide for that model.</p>")
        name, descale, clean = KELVIN_GUIDES[model]
        return self._kelvin(f"{name} ({model}): care and maintenance", f"""<h2>Care schedule</h2><table>
<tr><th>What</th><th>How often</th></tr><tr><td>Clean the filter</td><td>{clean}</td></tr>
<tr><td>Descale (Kelvin DescaleMax or citric acid)</td><td>{descale}</td></tr>
<tr><td>Wipe the door seal</td><td>every week</td></tr><tr><td>Check the spray arms turn freely</td><td>every month</td></tr>
</table><h2>Troubleshooting</h2><p><b>Rattling during the eco programme:</b> make sure the lower spray arm clicks into
place and nothing hangs below the bottom basket.</p><p><b>White film on glasses:</b> top up the rinse aid and descale.</p>
<p>Still need help? Call Kelvin Care on 1800 555 0199.</p>""")


KELVIN_NAV = (("", "Support"), ("products", "Products"), ("service", "Book a repair"), ("register", "Register a product"),
              ("warranty", "Warranty"))
KELVIN_GUIDES = {  # model: (name, descaling, filter)
    "DW-410": ("Series 2 dishwasher", "every 6 months", "every month"),
    "DW-412": ("Series 4 dishwasher", "every 3 months (every 2 months in hard-water areas)", "every 2 weeks"),
    "DW-414": ("Series 6 dishwasher", "every 4 months", "every 2 weeks"),
    "RF-260": ("FrostFree refrigerator 260 L", "not needed", "replace the water filter every 6 months"),
    "WM-7": ("Front-load washing machine 7 kg", "every 2 months (drum clean programme)", "every 3 months"),
}
LIBRARY_NAV = (("", "Home"), ("visit", "Visit us"), ("borrowing", "Borrowing"), ("events", "Events"),
               ("kids", "Kids & teens"), ("elibrary", "eLibrary"), ("support", "Support us"))
LIBRARY = {
    "": ("Welcome", """<form><label>Search the catalogue <input name="q"></label><button type="button">Search</button></form>
<h2>News</h2><article><h3>New reading room opens on the first floor</h3><p>Forty new seats, power at every desk and
quiet hours every morning.</p></article><article><h3>Summer reading challenge results</h3><p>1,240 children read
8,915 books this summer. Thank you to all our volunteers!</p></article><article><h3>Free coding club for teens</h3>
<p>Saturdays in the Makerspace. <a href="/library/events">See all events</a>.</p></article>
<h2>Quick links</h2><p><a href="/library/visit">Opening hours and directions</a> · <a href="/library/borrowing">Loans,
renewals and fees</a> · <a href="/library/elibrary">eBooks and audiobooks</a> · <a href="/library/support">Become a
member</a></p>"""),
    "visit": ("Visit us", """<p>Riverside Public Library, 14 Lake View Road, Riverside. Step-free entrance on Mill Lane.</p>
<h2>Opening hours</h2><table><tr><th>Day</th><th>Hours</th></tr><tr><td>Monday</td><td>Closed</td></tr>
<tr><td>Tuesday</td><td>09:00–20:00</td></tr><tr><td>Wednesday</td><td>09:00–20:00</td></tr>
<tr><td>Thursday</td><td>09:00–20:00</td></tr><tr><td>Friday</td><td>09:00–18:00</td></tr>
<tr><td>Saturday</td><td>09:00–17:00</td></tr><tr><td>Sunday</td><td>10:00–14:00</td></tr></table>
<p>The Makerspace opens one hour after the library and closes one hour before it.</p>
<h2>Getting here</h2><p>Buses 21 and 44 stop outside. Paid parking on Mill Lane; the first 30 minutes are free.</p>"""),
    "borrowing": ("Borrowing", """<p>Members can borrow up to 12 items at a time. Renew online, by phone or at the desk.</p>
<h2>Loan periods and late fees</h2><table><tr><th>Item</th><th>Loan period</th><th>Late fee per day</th></tr>
<tr><td>Books and audiobooks</td><td>21 days</td><td>₹5</td></tr><tr><td>Magazines</td><td>7 days</td><td>₹5</td></tr>
<tr><td>DVDs and Blu-ray discs</td><td>7 days</td><td>₹15</td></tr><tr><td>Board games</td><td>14 days</td><td>₹10</td></tr>
<tr><td>Laptops (in library only)</td><td>4 hours</td><td>₹50 per hour</td></tr></table>
<p>Late fees stop at ₹300 per item. Children's books have no late fees.</p>
<h2>Lost or damaged items</h2><p>We charge the replacement cost plus a ₹100 processing fee.</p>"""),
    "events": ("Events", """<article><h3>Teen coding club</h3><p>Saturdays 10:00–12:00, Makerspace.</p></article>
<article><h3>Author talk: Writing the river</h3><p>Thursday 6 November, 18:30, Reading room. Free, booking needed.</p></article>
<article><h3>Baby rhyme time</h3><p>Tuesdays and Fridays 10:30.</p></article>"""),
    "kids": ("Kids & teens", "<p>Story time, homework help and the summer reading challenge.</p>"),
    "elibrary": ("eLibrary", "<p>Borrow eBooks and audiobooks with your library card number and PIN.</p>"),
    "support": ("Support us", "<p>Membership is free for residents. Donations keep the Makerspace open.</p>"),
    "contact": ("Contact", "<p>Phone 0484 220 1150. Email desk@riverside-library.example.</p>"),
    "accessibility": ("Accessibility", "<p>Step-free access, a hearing loop at the desk and large-print books.</p>"),
}


# ---- the owner's documents --------------------------------------------------------------------------------

DOCUMENTS = {
    "Receipts/electra-invoice-2025-03-14.txt": """ELECTRA HOME STORE — TAX INVOICE
Invoice no. EHS/BLR/25/03/0871        Date: 14 March 2025
Store: Indiranagar, Bengaluru

1  Kelvin Series 4 Dishwasher DW-412 (silver)        ₹42,990
1  Electra Care+ extended warranty for the item above:
   3 years of cover after the manufacturer's warranty ends        ₹2,499
1  Installation kit                                          ₹650

Total (incl. GST)                                           ₹46,139
Paid by card ending 4421. Keep this invoice to claim Electra Care+.
Electra Care+ claims: call the Kelvin service line, quote this invoice number.
""",
    "Manuals/kelvin-dishwasher-warranty-card.txt": """KELVIN APPLIANCES — WARRANTY CARD
Product: Dishwasher, Series 4 (DW-412)

Kelvin warrants this appliance for 2 years from the date of purchase against defects in materials and
workmanship. Keep your purchase invoice: it is your proof of the purchase date.

Service and repairs: call Kelvin Care on 1800 555 0199 (toll free, 8 am to 8 pm, all days),
or book a visit at kelvin-appliances.example/service.
""",
    "Manuals/kelvin-fridge-warranty-card.txt": """KELVIN APPLIANCES — WARRANTY CARD
Product: FrostFree refrigerator, 260 L (RF-260)
Purchased: 2 January 2024

Comprehensive warranty: 1 year from the date of purchase. Compressor: 10 years.
Refrigerator service: call 1800 555 0177.
""",
    "Home/lease-2025.md": """# Flat lease — 3B, Lakeside Apartments
- Start: 1 June 2025, 24 months (ends 31 May 2027)
- Rent ₹38,000 a month, due by the 5th; deposit ₹1,50,000
- Notice period: 2 months
- Landlord: phone 415 555 0163
""",
    "Travel/goa-trip.md": """# Goa, 19–23 December 2025
- Flights booked: BLR → GOX 19 Dec 07:10, return GOX → BLR 23 Dec 18:40
- Stay: Casa Verde, Anjuna, 4 nights
- To pack: sunscreen, chargers, the blue backpack
""",
    "Notes/todo.md": """# To do
- Dishwasher installed 15 March 2025 — makes a rattling noise on the eco cycle, call service if it continues
- Renew car insurance before 10 November
- Return library DVDs
""",
}


def write_documents(folder: pathlib.Path) -> None:
    for name, text in DOCUMENTS.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
