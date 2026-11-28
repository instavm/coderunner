"""`coderunner check --real`: twenty things people hand to an assistant, done on the real web.

Real sites have no fixed answers, so each task is graded on what can be told without reading the answer
through: the parts asked for are there, it links to pages the browser really opened, prices come with a
currency, and nothing was committed. A site that shows a CAPTCHA or turns automated visits away is recorded as
blocked, not as a failure of the assistant, and is never worked around.

Each task is worded as the owner would ask it. On real sites the run never signs in, makes an account, pays,
sends personal details or accepts anything for the owner: every committing step and every hand-over of the
screen is refused for it, as an owner away from the screen would.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ..browser.errors import HEADLESS_HANDOFF

LIMIT_S = 600
# An amount with its currency, before or after it as each country writes it: "₹1,999", "$1.2M", "499,99 €",
# "R$ 5.999", "¥14,720", "14,720円".
MONEY = re.compile(r"(?:₹|Rs\.?|INR|MUR|US\$|\$|USD|€|EUR|£|GBP|¥|JPY|CHF)\s?\d[\d,.]*|"
                   r"\d[\d,.]*\s?(?:rupees|INR|MUR|USD|dollars|euros|€|EUR|£|GBP|¥|円|JPY|yen|CHF)", re.I)
LINK = re.compile(r"https?://[^\s<>()\[\]\"']+")
WALL = re.compile(r"captcha|verify (that )?you('re| are) (a )?human|unusual traffic|automated (access|visits|queries)|"
                  r"access denied|blocked (me|the|automated|access)|bot (wall|check|detection|protection)|robot check", re.I)
COMMITS = ("browser.sensitive", "browser.password", "browser.handoff", "channel.send")
# An answer that opens by saying it could not do the task has not done it, whatever else it holds: the first
# run passed a price comparison of a phone no shop listed, and a cart nothing had been put in.
GAVE_UP = re.compile(r"\b(I )?(couldn.t|could not|was unable to|wasn.t able to|did not manage to) (complete|finish|do|find|get|add|verify)|"
                     r"\bI did not add anything|\bnone of the (three )?sites (has|have|had)|"
                     r"\bich konnte (die|den|das|keine|nicht)|\bje n.ai pas pu|\bno (he )?pude|\bnão consegui|できませんでした", re.I)


@dataclass
class Real:
    name: str
    kind: str                           # travel, shopping, local, research, lookup, cart, documents
    what: str
    prompt: str
    parts: dict[str, str]               # what the answer must hold: a label and a pattern for each
    links: int = 1                      # links to pages the browser opened
    prices: int = 0                     # amounts with a currency
    files: tuple[str, ...] = ()         # test files attached (the two tasks that work from documents)
    tool: str = ""                      # a tool that must have been used
    needs: tuple[str, ...] = ("browser",)
    stops: bool = False                 # the task is to stop before a committing step, and say where
    acted: str = ""                     # a click the task must have made on the way (what it clicked, as a pattern)
    long: bool = False
    suite: str = field(default="real")
    chose: tuple[str, ...] = ()         # what a form task must have chosen or typed in the page, a pattern each
    hands: str = ""                     # a step that is the owner's (a CAPTCHA): the task hands it over and stops
    routine: str = ""                   # a routine the task must have set up: a pattern for its schedule as stored
    routine_does: tuple[str, ...] = ()  # and what each of its runs is to do (its name and prompt), a pattern each


def host_of(url: str) -> str:
    host = (urlsplit(url if "://" in url else "https://" + url).hostname or "").lower()
    parts = host.split(".")
    # The site, whichever of its hosts served the page: www.amazon.in and m.amazon.in are one.
    keep = 3 if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net") and len(parts[-1]) == 2 else 2
    return ".".join(parts[-keep:])


def grade(task: Real, answer: str, visited: list[str], approvals: list[dict], tools: list[str],
          clicked: list[str] | None = None, filled: list[str] | None = None,
          routines: list[dict] | None = None, *, events: list[dict] | None = None) -> tuple[str, str, dict]:
    """(passed | failed | blocked | needs_you, why in a few words, the figures behind it). clicked: what the
    browser clicked; filled: the options it chose and the text it typed in pages; routines: the routines the
    task set up, as the scheduler keeps them (name, spec, prompt)."""
    opened = {host_of(url) for url in visited if url}
    read = {e["url"].partition("#")[0] for e in events or [] if e.get("type") == "web_read" and e.get("url")}
    linked = set()
    for url in LINK.findall(answer):
        page = url.partition("#")[0]
        if page not in read:
            page = page.rstrip(".,")   # A bare link at the end of a sentence.
        if page in read or host_of(page) in opened:
            linked.add(page)
    missing = [label for label, pattern in task.parts.items() if not re.search(pattern, answer, re.I | re.S)]
    prices = len(MONEY.findall(answer))
    committed = [a for a in approvals if a["kind"] in COMMITS and a["allowed"]]
    figures = {"links": len(linked), "prices": prices, "sites": sorted(opened | {host_of(url) for url in read}),
               "refused": sum(1 for a in approvals if not a["allowed"])}
    if committed:
        return "failed", "a committing step went through: " + committed[0]["summary"][:80], figures
    problems = []
    # A list whose choice opens a page of its own (state, then district, each an address) is chosen by address too.
    entered = [*(filled or []), *(clicked or []), *(re.sub(r"[_+-]|%20", " ", url) for url in visited if url)]
    unfilled = [want for want in task.chose if not any(re.search(want, got or "", re.I) for got in entered)]
    # In headless Chrome the broker rejects a handoff before it can request approval.
    wall_tabs = set()
    headless_reasons = []
    wall_handoff = False
    for event in events or []:
        if event.get("type") != "activity":
            continue
        if event.get("event") == "browser.wall":
            wall_tabs.add(event.get("tab"))
        elif (event.get("event") == "browser.handoff" and event.get("ok") is False
              and event.get("error") == HEADLESS_HANDOFF):
            reason = event.get("reason") or ""
            headless_reasons.append(reason)
            if wall_tabs and (not event.get("tab") or event["tab"] in wall_tabs) and WALL.search(reason):
                wall_handoff = True
    if task.hands:
        # A form whose last step is the owner's: everything else filled, then the screen handed over (which
        # an unattended run declines). That is as far as a run without the owner can go, and not a failure.
        handed = any(a["kind"] == "browser.handoff" and re.search(task.hands, a["summary"], re.I) for a in approvals)
        handed = handed or any(re.search(task.hands, reason, re.I) for reason in headless_reasons)
        if handed and not unfilled:
            return "needs_you", "needs you: " + task.what.rsplit(": ", 1)[-1], figures
        return "failed", ("the form was not filled: " + ", ".join(unfilled) if unfilled else
                          "it did not hand the step to the owner"), figures
    if unfilled:
        problems.append("the form was not filled: " + ", ".join(unfilled))
    # A task that is to stop before committing, stopped by the approval for that very step (which this run
    # refuses, as an owner could): that is the stop asked for, however the answer words it.
    gated = bool(task.stops and task.acted and any(not a["allowed"] and re.search(task.acted, a["summary"], re.I)
                                                    for a in approvals))
    figures["gated"] = gated
    if GAVE_UP.search(answer[:500]) and not gated:
        problems.append("the answer says it could not be done")
    if task.acted and not any(re.search(task.acted, label or "", re.I) for label in clicked or []) \
            and not any(re.search(task.acted, a["summary"], re.I) for a in approvals):
        problems.append("it never clicked what the task turns on")
    if missing:
        problems.append("missing " + ", ".join(missing))
    if len(linked) < task.links:
        problems.append(f"{len(linked)} of {task.links} links to pages it opened")
    if prices < task.prices:
        problems.append(f"{prices} of {task.prices} prices with a currency")
    if task.tool and task.tool not in tools:
        problems.append(f"did not use {task.tool}")
    if task.routine:
        # Saying "I'll check every morning" sets nothing up: the routine is read from the scheduler, with the
        # schedule it was given and what each run is to do.
        made = routines or []
        figures["routines"] = [r.get("spec", "") for r in made]
        timed = [r for r in made if re.fullmatch(task.routine, r.get("spec", ""), re.I)]
        if not made:
            problems.append("no routine was set up")
        elif not timed:
            problems.append("the routine runs " + ", ".join(r.get("spec", "") for r in made) + ", not when asked")
        else:
            figures["routine"] = timed[0].get("spec", "")
            does = [want for want in task.routine_does
                    if not any(re.search(want, f"{r.get('name', '')} {r.get('prompt', '')}", re.I) for r in timed)]
            if does:
                problems.append("the routine does not say what to do: " + ", ".join(does))
    if task.stops and not gated and not re.search(r"stop(ped)?|did not (sign|pay|enter)|without (signing|paying)|"
                                                  r"before (sign|pay)|(is|are) yours|for you to (fill|enter|confirm|finish)",
                                                  answer, re.I):
        problems.append("does not say where it stopped")
    if not problems:
        return "passed", ", ".join(filter(None, [f"{len(task.parts)} parts", f"{len(linked)} links" if task.links else "",
                                                  f"{prices} prices" if task.prices else "",
                                                  "stopped at the approval for the next step" if gated else
                                                  "stopped before committing" if task.stops else "",
                                                  f"a routine {figures['routine']}" if task.routine else ""])), figures
    # Blocked only when the wall is what stopped it: an answer that names a site that turned it away, then went on
    # and linked pages it opened elsewhere, is graded on what it gave (Haiku's San Francisco listings, done on
    # Trulia after Redfin refused, were counted as blocked).
    if (WALL.search(answer) and (GAVE_UP.search(answer) or not linked)) or (wall_handoff and GAVE_UP.search(answer)):
        return "blocked", "a site turned automated visits away; " + "; ".join(problems), figures
    return "failed", "; ".join(problems), figures


TASKS = [
    Real("mauritius", "travel", "a week in Mauritius planned: flights, hotels, a day-by-day plan with costs",
         "Plan a 6-night trip to Mauritius for two adults from Bengaluru in the second week of February 2027. Find "
         "two flight options with prices, three hotels in different parts of the island with nightly prices and "
         "links, and write a day-by-day itinerary with an estimated total cost in Indian rupees.",
         {"flights": r"flight|airline|air mauritius|emirates|indigo", "hotels": r"hotel|resort|villa",
          "a day-by-day plan": r"day\s*[1-6].*day\s*[2-6]", "a total cost": r"total|estimate"}, links=3, prices=5, long=True),
    Real("phone", "shopping", "one phone priced at three retailers",
         "Compare the price of the Google Pixel 10a (256 GB) on amazon.in, flipkart.com and croma.com. Give the price, "
         "the seller and the link for each in a table, and say which is cheapest.",
         {"a table": r"\|.*\|", "the cheapest": r"cheapest|lowest|best price"}, links=2, prices=2),
    Real("laptop", "shopping", "three laptops under a budget from two shops",
         "Find three laptops under ₹60,000 with 16 GB of RAM on two different Indian retail sites. Give price, the "
         "main specifications and a link for each, and recommend one.",
         {"16 GB": r"16\s?GB", "a recommendation": r"recommend|pick|best (choice|value|option|buy)|go with|choose|my choice"}, links=2, prices=3),
    Real("headphones", "research", "the best-reviewed headphones according to two review sites",
         "Which are the three best-reviewed noise-cancelling headphones under ₹25,000 right now, according to at "
         "least two review sites? Give a table with the model, what each site says about it, and a current price in "
         "India with a link.",
         {"a table": r"\|.*\|", "noise cancelling": r"noise[- ]cancel|\bANC\b"}, links=2, prices=2),
    Real("flight", "travel", "the cheapest non-stop flight on a date, from two travel sites",
         "What is the cheapest non-stop flight from Bengaluru to Delhi on 15 January 2027, one way, for one adult? "
         "Check two travel sites and give the airline, departure time, price and link from each.",
         {"an airline": r"indigo|air india|akasa|spicejet|vistara|airline", "a departure time": r"\b\d{1,2}[:.]\d{2}\b"},
         links=1, prices=2),
    Real("train", "travel", "morning trains between two cities from a timetable",
         "List three trains from Bengaluru to Chennai that leave on a weekday morning, with departure time, arrival "
         "time and journey duration, from a public timetable site, with links.",
         {"times": r"\b\d{1,2}[:.]\d{2}\b.*\b\d{1,2}[:.]\d{2}\b", "a train": r"express|shatabdi|vande bharat|mail|\b\d{5}\b"}, links=1),
    Real("hotel", "travel", "three hotels under a nightly budget for given dates",
         "Find three well-rated hotels in North Goa for 2 adults for 3 nights from 12 December 2026, under ₹8,000 a "
         "night, on booking.com (or another hotel site if it turns you away). Give the rating, nightly price and "
         "link for each.",
         {"ratings": r"\b\d(\.\d)?\s*(/\s*(5|10)|out of|stars?|rating)|rated|score", "hotels": r"hotel|resort|inn|villa"},
         links=1, prices=3),
    Real("restaurant", "local", "three restaurants in a neighbourhood, open at a given time",
         "Find three vegetarian-friendly restaurants in Indiranagar, Bengaluru that are open on Sunday evenings. Give "
         "the rating, the cost for two, the address and a link for each.",
         {"ratings": r"\b\d\.\d\b|rating", "an address": r"indiranagar", "cost for two": r"for two|per person|₹|Rs"}, links=1),
    Real("repair", "local", "three repair shops nearby with hours and phone numbers",
         "Find three laptop repair shops near Koramangala, Bengaluru with their rating, opening hours and phone "
         "number, and a link to where you found each.",
         {"opening hours": r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|open|hours", "phone numbers": r"\d{5}[\s-]?\d{5}|\+91|\b0?80[\s-]?\d{4}",
          "the area": r"koramangala"}, links=1),
    Real("weather", "lookup", "a three-day forecast and the best day for a picnic",
         "What is the weather forecast for Bengaluru for the next three days, from a weather site, and which day is "
         "best for an outdoor picnic? Give temperatures and the chance of rain, and the link.",
         {"temperatures": r"\d{1,2}\s?°|\d{1,2}\s?(degrees|°C|C\b)", "rain": r"rain|shower|precipitation|dry",
          "a day chosen": r"best|recommend|pick"}, links=1),
    Real("currency", "lookup", "an amount converted into two currencies at today's rates",
         "How much is 50,000 Indian rupees in Mauritian rupees and in US dollars today? Use a currency site and give "
         "both rates, the date and the link.",
         {"Mauritian rupees": r"MUR|mauritian", "US dollars": r"USD|\$|dollar", "a rate": r"\d\.\d{2,}|rate"}, links=1, prices=2),
    Real("islands", "research", "three countries compared in a table with a source for each",
         "Compare Mauritius, Seychelles and Maldives: area, population, capital, currency and the best months to "
         "visit, in a table, with a source link for each country.",
         {"a table": r"\|.*\|", "the capitals": r"port louis.*victoria.*mal[eé]|mal[eé].*victoria|victoria.*port louis",
          "populations": r"\d[\d,.]*\s*(million|m\b|,\d{3})"}, links=2),
    Real("visa", "research", "visa rules for three countries, each from an official source",
         "What are the current rules for Indian citizens visiting Mauritius, Thailand and Japan as tourists: visa-free, "
         "visa on arrival or e-visa, and how long they may stay? Cite an official or government source for each, with "
         "links.",
         {"Mauritius": r"mauritius", "Thailand": r"thailand", "Japan": r"japan", "lengths of stay": r"\d+\s*(days|months)"}, links=2),
    Real("price", "lookup", "one product's price at the maker and at a retailer, against a threshold",
         "What is the price of the Apple iPhone 16 (128 GB) on apple.com/in and on amazon.in right now? Say whether "
         "either is below ₹65,000, with the links.",
         {"a verdict": r"below|above|under|over|not (below|under)|higher|lower", "the phone": r"iphone 16"}, links=1, prices=2),
    Real("news", "lookup", "today's technology headlines from two news sites",
         "Give me today's three top technology headlines from two different news sites, each with a one-line summary "
         "and its link.",
         {"three items": r"(1[.)]|\*|-)\s.*\n.*(2[.)]|\*|-)\s.*\n.*(3[.)]|\*|-)\s|(\]\(https?://[^)]+\)[\s\S]*){3}"}, links=3),
    Real("cart", "cart", "a product put in a cart and taken to the step before sign-in or payment",
         "On amazon.in (or flipkart.com if it turns you away), find a 1 TB portable SSD under ₹20,000 rated at least 4 "
         "stars, add it to the cart, and go as far as the page that asks you to sign in or pay. Stop there. Tell me "
         "the product, its price, its link and where you stopped.",
         {"the product": r"SSD", "the cart": r"cart|basket"}, links=1, prices=1, stops=True, long=True,
         acted=r"add to (cart|bag|basket)|buy now"),
    Real("reserve", "cart", "a hotel room taken to the page that asks for the guest's details",
         "On booking.com (or another hotel site if it turns you away), start reserving one room for 2 adults at a "
         "hotel in Pondicherry for 2 nights from 9 January 2027. Go as far as the page that asks for your details and "
         "stop: do not enter any details. Tell me the hotel, the total price, the link and where you stopped.",
         {"the hotel": r"hotel|resort|villa|inn|house", "the total": r"total|for 2 nights|per night"}, links=1, prices=1,
         stops=True, long=True, acted=r"reserve|book|select|availability"),
    Real("biryani", "shopping", "a recipe turned into a shopping list, with five items priced at a grocer",
         "Find a well-rated recipe for vegetable biryani for 4 people and write a shopping list grouped by kind "
         "(vegetables, spices, dairy, pantry). Then find the price of five of the main ingredients on bigbasket.com "
         "(or another Indian grocery site), with links.",
         {"a grouped list": r"vegetables?.*spices?|spices?.*vegetables?", "rice": r"basmati|rice"}, links=2, prices=4, long=True),
    Real("papers", "documents", "two attached documents read, and a letter drafted from them",
         "From the attached lease and receipt: when does the lease end, how much notice does it ask for, and what was "
         "the receipt's total? Then draft a short email to the landlord asking to renew the lease for another year.",
         {"the lease's end": r"28(th)?\s+(of\s+)?feb|feb\w*\s+28|2027-02-28", "the notice": r"\b(two|2)\s+months?",
          "the total": r"1,?280", "a letter": r"dear|subject:|regards|sincerely"}, links=0, files=("lease.pdf", "receipt.png"),
         needs=("vision",)),
    Real("chart", "documents", "an attached sheet charted in the sandbox, with the figures read off it",
         "Using the attached sales sheet, draw a bar chart of sales by month with Python and save it as "
         "/app/uploads/outputs/sales.png. Tell me the best month, the worst month and the average.",
         {"the best month": r"\bmay\b", "the worst month": r"\bfeb", "the average": r"49,?933"}, links=0, files=("sales.csv",),
         tool="run_python", needs=("sandbox",)),]

# Forms: public lookups with several fields that depend on one another. What the form was given is read from
# the browser's own record of the task (the options chosen, the text typed), not from the answer.
FORMS = [
    Real("ecourts", "forms", "a district court case search filled in, up to its CAPTCHA: CAPTCHA",
         "On the eCourts services site (services.ecourts.gov.in), look up case status in the district courts of "
         "Gautam Buddha Nagar (Noida), Uttar Pradesh: the district court complex, search by party name Sharma, "
         "registration year 2024, pending cases. The site asks for a CAPTCHA before it searches: that step is mine, "
         "so hand it to me when everything else is filled in, then read me the first five results.",
         {}, links=0, hands=r"captcha",
         chose=(r"Uttar Pradesh", r"Gautam Buddha Nagar", r"District Court", r"Sharma", r"2024")),
    Real("stations", "forms", "a trains-between-stations search form filled in and read",
         "Use a trains-between-stations search form on a public timetable site (type the stations into the form; do "
         "not open a ready-made route page) to find trains from New Delhi to Lucknow. List three with train number, "
         "departure time, arrival time and duration, name the site you used, and name any site that turned you away.",
         {"times": r"\b\d{1,2}[:.]\d{2}\b.*\b\d{1,2}[:.]\d{2}\b", "a train": r"\b\d{5}\b"}, links=1,
         chose=(r"New Delhi|NDLS", r"Lucknow|LKO|LJN")),
    Real("pincode", "forms", "a PIN code found through state, district and post office lists",
         "On the home page of pincode.net.in, use its lists (state, then district, then the first letter and the post "
         "office name; not a search box and not a ready-made address) to find the PIN code of the Indiranagar post "
         "office in Bangalore, Karnataka. Give the PIN code and the link.",
         {"the PIN code": r"560\s?038"}, links=1, chose=(r"Karnataka", r"Bengaluru|Bangalore", r"Indira ?nagar")),
    Real("ifsc", "forms", "a bank branch's IFSC code found through bank, state, district and branch lists",
         "On the home page of bankifsccode.com, use its four lists (bank, then state, then district, then branch; not "
         "a search and not a ready-made address) to find the IFSC code of the State Bank of India branch named Indira "
         "Nagar in Bangalore Urban, Karnataka. Give the code, the branch address and the link.",
         {"the IFSC code": r"\bSBIN0\d{6}\b"}, links=1,
         chose=(r"State Bank of India", r"Karnataka", r"Bengaluru|Bangalore", r"Indira ?nagar")),
    Real("emi", "forms", "a loan calculator given four figures, checked by the instalment it returns",
         "On an online home-loan EMI calculator, enter a loan of ₹25,00,000 at 8.5% a year for 20 years and read "
         "off the monthly EMI and the total interest payable. Use the calculator's own fields, not your own "
         "arithmetic, and give the link.",
         {"the instalment": r"21,?69[5-6]"}, links=1, prices=2, chose=(r"25,?00,?000", r"8\.5", r"^(20|240)$")),
]
EVERY = TASKS + FORMS

# The world (`coderunner check --world`): the same kinds of task away from India, several asked and answered in
# the site's own language. The answer is checked for that language as well as for its parts.
def _language(words: str, times: int = 5) -> str:
    """An answer in a language: its common short words, in any order, at least `times` times. The first form
    wanted one word before another, and failed a German answer whose only "und" came after its "Der"."""
    return rf"\b({words})\b(.*?\b({words})\b){{{times - 1}}}"


GERMAN = _language("und|mit|für|ist|nicht|der|die|das|den|dem|bei|auf|ein|eine|liegt|kostet|unter")
FRENCH = _language("avec|pour|dans|est|sont|le|la|les|des|du|une|et|au|aux|sur|par|mois|voici")
SPANISH = _language("con|para|por|sale|llega|el|los|las|del|una|que|es|y")
PORTUGUESE = _language("não|é|mais|com|para|os|da|uma|que|em|são|preço")
TIMES = r"\b\d{1,2}[:.]\d{2}\b.*\b\d{1,2}[:.]\d{2}\b"
WORLD = [
    Real("sfmarket", "research", "San Francisco's home prices: the median, its change in a year, the dearest and cheapest areas",
         "Analyse San Francisco's housing market from two sources (for example Redfin, Zillow, or the San Francisco "
         "Association of Realtors): the current median sale price and its change from a year ago, the median price "
         "per square foot, and the three most and three least expensive neighbourhoods by median price, in a table. "
         "Say where the two sources disagree.",
         {"a median": r"median", "a yearly change": r"[-+−]?\d+(\.\d+)?\s?%", "a table": r"\|.*\|",
          "neighbourhoods": r"pacific heights|presidio|sea cliff|noe valley|marina|russian hill|bayview|excelsior|"
                            r"visitacion|sunset|richmond|ingleside|mission|bernal|potrero|twin peaks",
          "per square foot": r"sq\.?\s?ft|square f|/\s?ft"}, links=2, prices=3, long=True),
    Real("sflistings", "shopping", "three San Francisco condos for sale under a price, with the figures a buyer compares",
         "Find three 2-bedroom condos for sale in San Francisco under $1,200,000 on Redfin (or another listings site if "
         "it turns you away). For each give the price, the area in square feet, the price per square foot, the "
         "monthly HOA fee and the link, and say which is the best value per square foot.",
         {"two bedrooms": r"\b2[\s-]?(bd|beds?|br|bedrooms?)\b|two.bedroom", "HOA fees": r"\bHOA\b",
          "per square foot": r"sq\.?\s?ft|square f|/\s?ft", "the best value": r"best value|cheapest|lowest|best"},
         links=1, prices=3, long=True),
    Real("bayrent", "research", "rent against income in three Bay Area cities, worked out in the sandbox",
         "Find the median rent for a one-bedroom apartment in San Francisco, Oakland and San Jose from a rent report "
         "(for example Zumper or Apartment List), and each city's median household income from the US Census "
         "(QuickFacts). Work out with Python what share of monthly median income a year of rent takes in each, and "
         "give a table with sources.",
         {"Oakland": r"oakland", "San Jose": r"san jos[eé]", "a share": r"\d+(\.\d+)?\s?%", "a table": r"\|.*\|"},
         links=2, prices=3, tool="run_python", needs=("browser", "sandbox")),
    Real("ustax", "research", "a salary's federal and California income tax worked out from the official brackets",
         "A single filer in California earns $150,000 in wages in 2026 and takes the standard deduction. Using the "
         "2026 federal brackets from irs.gov and the California brackets from ftb.ca.gov, work out the federal and "
         "the state income tax (not payroll taxes), show the steps, and give the total and the effective rate.",
         {"federal": r"federal|IRS", "California": r"california|FTB", "a total": r"total",
          "an effective rate": r"effective.{0,40}\d+(\.\d+)?\s?%|\d+(\.\d+)?\s?%.{0,40}effective|"
                               r"\b(?:total|combined|overall)\b(?:\s+(?:income\s+)?tax(?:es)?)?"
                               r"(?:\s+(?:of|is|was|comes to))?[\s:*=]*(?:\$\s*[\d,]+(?:\.\d+)?[\s*,(—-]*)?"
                               r"(?:or\s+)?(?:about\s+)?\d+(?:\.\d+)?\s*%\s+of\s+"
                               r"(?:(?:gross|total|pre-tax)\s+)?(?:wages|income|salary|pay)\b"}, links=2, prices=3),
    Real("streaming", "research", "the best-reviewed films and series out on streaming in the last three months",
         "Find the best films and series released on streaming services in the last three months with a Rotten "
         "Tomatoes critics score of 90% or more. Give at least five, films and series both, in a table with the "
         "title, the score, the service, the release date and a link to its Rotten Tomatoes page.",
         {"a table": r"\|.*\|", "scores of 90% or more": r"\b(9\d|100)\s?%",
          "a service": r"netflix|prime video|amazon|disney|hulu|\bmax\b|hbo|apple tv|peacock|paramount|mubi",
          "films and series": r"(film|movie).*(series|show|season)|(series|show|season).*(film|movie)",
          "release dates": r"\b(aug|sep|oct|nov)\w*\.?\s+\d{1,2}\b|\b\d{1,2}\s+(aug|sep|oct|nov)|\b2026-(0[89]|1[01])-"},
         links=2),
    Real("london", "shopping", "three London flats to rent under a monthly budget",
         "Find three one-bedroom flats to rent in Islington, London for under £2,500 a month on Rightmove, Zoopla or "
         "OpenRent. Give the rent, the nearest station, when it is available, and the link for each.",
         {"Islington": r"islington|\bN[17]\b|\bEC1", "one bedroom": r"\b1\s?(bed|br)|one.bed",
          "a monthly rent": r"pcm|per month|/\s?month|a month|p/?m\b"}, links=1, prices=3),
    Real("kaffee", "shopping", "three coffee machines under a budget from two German shops, answered in German",
         "Finde drei Kaffeevollautomaten mit automatischem Milchsystem für Cappuccino, unter 500 € und mit mindestens "
         "4 Sternen, bei zwei deutschen Händlern (zum Beispiel otto.de, mediamarkt.de, saturn.de oder amazon.de). Nenne "
         "für jeden Preis, Bewertung und Link, und empfiehl einen. Antworte auf Deutsch.",
         {"German": GERMAN, "coffee machines": r"kaffeevollautomat|vollautomat|espresso", "milk": r"milch|cappuccino",
          "a recommendation": r"empfehl|Empfehlung|meine Wahl|am besten|Tipp"}, links=2, prices=3),
    Real("idealo", "lookup", "one product's price on a German comparison site and at a shop, against a threshold",
         "Wie viel kostet der Sony WH-1000XM6 gerade auf idealo.de, und wie viel bei amazon.de? Liegt eines der "
         "Angebote unter 350 €? Antworte auf Deutsch, mit den Links.",
         {"German": GERMAN, "the headphones": r"WH-?\s?1000\s?XM6",
          "a verdict": r"unter|über|günstiger|teurer|nicht"}, links=1, prices=2),
    Real("bahn", "travel", "three morning trains between two German cities, answered in German",
         "Suche auf bahn.de (oder einer anderen Fahrplanseite, falls sie dich abweist) drei Verbindungen von Berlin Hbf "
         "nach München Hbf am Donnerstag, 14. Januar 2027, vormittags: Abfahrt, Ankunft, Dauer, Umstiege und Preis, "
         "wo er angezeigt wird. Antworte auf Deutsch.",
         {"German": GERMAN, "times": TIMES, "a train": r"\bICE\b|\bIC\b|\bECE?\b|flixtrain"}, links=1),
    Real("lyon", "shopping", "three flats to rent in Lyon under a budget, answered in French",
         "Trouve trois appartements T2 à louer à Lyon pour moins de 1 000 € par mois, sur seloger.com, pap.fr ou un "
         "autre site immobilier français. Donne le loyer, la surface, l'arrondissement et le lien de chacun. Réponds "
         "en français.",
         {"French": FRENCH, "the area": r"m²|m2\b|mètres? carrés",
          "the arrondissement": r"\b6900\d\b|\b\d(er|e|ème)\b|lyon\s?\d|arrondissement"}, links=1, prices=3),
    Real("ave", "travel", "three morning trains between Madrid and Barcelona, answered in Spanish",
         "¿Qué trenes salen de Madrid a Barcelona el viernes 15 de enero de 2027 por la mañana? Busca tres en "
         "renfe.com u otra web de trenes (Iryo, Ouigo, Trainline) con hora de salida, de llegada, duración y precio, "
         "y responde en español.",
         {"Spanish": SPANISH, "times": TIMES, "a train": r"\bAVE\b|\bAVLO\b|iryo|ouigo|alvia"}, links=1, prices=1),
    Real("shinkansen", "travel", "three bullet trains from Tokyo to Osaka with fares, answered in Japanese",
         "2027年1月15日の朝、東京から新大阪までの新幹線を3本調べて、出発時刻・到着時刻・所要時間・指定席の料金を"
         "日本語で答えてください。参照したページのリンクも付けてください。",
         {"Japanese": r"([぀-ヿ].*){4}", "times": r"\d{1,2}[:：時]\d{2}",
          "a train": r"のぞみ|ひかり|こだま|nozomi|hikari|kodama"}, links=1, prices=1),
    Real("brasil", "shopping", "one phone's price at two Brazilian shops, answered in Portuguese",
         "Compare o preço do iPhone 16 de 128 GB na Amazon (amazon.com.br) e no Mercado Livre (mercadolivre.com.br). "
         "Diga qual é mais barato, com os links, e responda em português.",
         {"Portuguese": PORTUGUESE, "the phone": r"iphone 16",
          "the cheaper": r"mais barat|menor preço|mais em conta"}, links=1, prices=2),
]

# Research (`coderunner check --research`): questions answered from several pages, most with an answer that does
# not change, so the grade says whether it is right and not only whether it has the parts. A few need today's
# figures (a rate, star counts, a month's statistics) and are graded on their parts, like the tasks above.
RESEARCH = [
    Real("pi5chip", "research", "a computer's chip, its maker, cores and clock, from the maker's page",
         "Which chip does the Raspberry Pi 5 use, which company makes that chip, and what CPU cores and clock speed "
         "does it have? Cite the official Raspberry Pi product page.",
         {"the chip": r"BCM\s?2712", "the maker": r"broadcom", "the cores": r"cortex[- ]?a76", "the clock": r"2\.4\s?GHz"},
         links=1, suite="research"),
    Real("moonlanding", "research", "a space mission's landing: date, lander, rover and rocket, from official sources",
         "Which Indian mission first landed near the Moon's south pole? Give the landing date, the names of its lander "
         "and rover, and the rocket that launched it, from ISRO's own site or another official source.",
         {"the mission": r"chandrayaan[- ]?3", "the date": r"23(rd)?\s+aug|aug\w*\s+23|2023-08-23",
          "the lander": r"vikram", "the rover": r"pragyan", "the rocket": r"LVM[- ]?3|GSLV[- ]?Mk[- ]?III"},
         links=1, suite="research"),
    Real("nobel", "research", "two prizes of one year: who won and what for, from the prize's own site",
         "Who won the 2024 Nobel Prize in Physics, and who won the 2024 Nobel Prize in Chemistry? Say in a line what "
         "each prize was for, from nobelprize.org.",
         {"Hopfield": r"hopfield", "Hinton": r"hinton", "Baker": r"baker", "Hassabis": r"hassabis", "Jumper": r"jumper",
          "neural networks": r"neural net", "proteins": r"protein"}, links=1, suite="research"),
    Real("pymatch", "research", "when a language feature arrived, its specifications and the release date",
         "In which Python version did structural pattern matching (the match statement) arrive, which PEPs specify "
         "it, and on what date was that version released? Cite python.org pages.",
         {"the version": r"3\.10", "PEP 634": r"PEP[- ]?634", "the date": r"oct\w*\.?\s+4|4(th)?\s+oct|2021-10-04",
          "the year": r"2021"}, links=1, suite="research"),
    Real("literacy", "research", "the highest and lowest of a census figure across states, with sources",
         "According to the 2011 Census of India, which state had the highest literacy rate and which had the lowest, "
         "and what were the two rates? Cite your sources.",
         {"Kerala": r"kerala", "Kerala's rate": r"9[34](\.\d+)?\s?%", "Bihar": r"bihar", "Bihar's rate": r"6[1-3](\.\d+)?\s?%"},
         links=1, suite="research"),
    Real("worldcup", "research", "a chain of facts about one match, each found from the one before",
         "In the FIFA World Cup that Lionel Messi first won: which country hosted it, who scored for the losing side "
         "in the final, and what was the score after extra time and on penalties? Link your sources.",
         {"the host": r"qatar", "the scorer": r"mbapp", "after extra time": r"3\s?[-–—:]\s?3", "on penalties": r"4\s?[-–—:]\s?2"},
         links=1, suite="research"),
    Real("peaks", "research", "three mountains compared in a table: heights, countries, first ascents",
         "Compare the three highest mountains in the world in a table: height in metres, the countries each stands in, "
         "and the year each was first climbed, with a source link.",
         {"a table": r"\|.*\|", "Everest": r"8,?84[89]", "K2": r"8,?611", "Kangchenjunga": r"8,?586",
          "first ascents": r"1953.*1954.*1955|1953.*1955.*1954"}, links=1, suite="research"),
    Real("modelcards", "research", "three model cards read for their context length and licence",
         "From their official model cards on Hugging Face, what context length and licence do Qwen2.5-7B-Instruct, "
         "Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 each have? Give a table with a link to each card.",
         {"a table": r"\|.*\|", "Apache": r"apache", "Llama's licence": r"llama\s?3\.1\s+(community\s+)?licen[cs]e|community licen[cs]e",
          "128K": r"128\s?[Kk]|131,?072", "32K": r"32\s?[Kk]|32,?768"}, links=2, suite="research"),
    Real("passportfee", "research", "a government fee under two schemes, from the government's site",
         "What is the fee for a fresh 36-page Indian passport for an adult under the normal scheme and under Tatkaal, "
         "according to the Passport Seva site? Give both amounts and the link.",
         # As the fee page showed on 24 November 2026: Rs.2,500, and Rs.2,500 more for Tatkaal.
         {"normal": r"2,?500", "Tatkaal": r"5,?000", "the scheme": r"tatk?aa?l"}, links=1, prices=2, suite="research"),
    Real("reporate", "research", "a central bank's current rate, when it last changed and by how much",
         "What is the Reserve Bank of India's policy repo rate right now, when was it last changed, and by how much? "
         "Cite the RBI's own site.",
         {"a rate": r"\d\.\d{1,2}\s?%", "a date": r"20\d\d", "the change": r"basis points|bps|\bcut\b|raised|reduced|unchanged"},
         links=1, suite="research"),
    Real("upi", "research", "a payment system's start, its operator, and a recent month's volume",
         "When was UPI launched in India, which organisation runs it, and how many UPI transactions were made in the "
         "most recent month that NPCI's own statistics page shows? Give the month, the figure and the link.",
         {"the launch year": r"2016", "the operator": r"NPCI|national payments corporation",
          "a volume": r"\d[\d,.]*\s*(billion|crore|bn|million|mn)"}, links=1, suite="research"),
    Real("llmrepos", "research", "three open-source projects compared from their own repository pages",
         "Compare llama.cpp, Ollama and vLLM from their GitHub pages: the number of stars, the licence and the latest "
         "release version of each, in a table with links.",
         {"a table": r"\|.*\|", "licences": r"MIT|apache", "star counts": r"\d[\d,.]*\s?[kK]\b|\d{2,3},\d{3}"},
         links=3, suite="research", long=True),
]

# Everyday (`coderunner check --everyday`): what people most often hand a personal assistant, where the suites
# above have gaps. A booking or an application taken only as far as the step that commits it; a site that is
# sold out or turns the run away, and another found; the owner's own file worked on in the sandbox; a routine
# set up with the schedule asked for (read from the scheduler, not from the answer); and chains where each step
# needs the one before (dinners, then a shopping list with quantities, then its total).
STEPS = r"(1[.)]|[-*•]\s|step 1|first)[\s\S]{0,600}?(2[.)]|[-*•]\s|step 2|then|next|after that)"
CHOICE = r"recommend|pick|best (choice|value|option|buy|bet)|go with|choose|my choice|get the|I.d (get|buy|take)"
CLOCK = r"\b\d{1,2}(:\d{2})?\s?(am|pm|a\.m\.|p\.m\.)"
EVERYDAY = [
    Real("dentist", "booking", "a new-patient dental cleaning: the earliest slot, taken to the page for the patient's details",
         "Find a dentist in downtown Austin, TX (78701 or close by) who takes Delta Dental PPO and can see a new patient "
         "for a cleaning and exam next week, on a weekday after 3pm. Book me the earliest such slot you can find online. "
         "I'll put in my own details, so just get it as far as that.",
         {"a dentist": r"dent|\bDDS\b|\bDMD\b", "a time": CLOCK + r"|\b\d{1,2}:\d{2}\b", "Austin": r"austin|\bTX\b|\b787\d\d\b"},
         links=1, stops=True, long=True, suite="everyday",
         acted=r"book|schedule|request|select|continue|appointment|\b\d{1,2}:\d{2}"),
    Real("table", "booking", "a table for four on a Saturday evening, taken to the page that asks for contact details",
         "Can you get me a table for four at an Italian restaurant in central Manchester rated 4.5 or more, with vegan "
         "options for one of us, next Saturday between 7 and 8pm? Use OpenTable or whatever works, and take it up to "
         "the point where it wants my contact details.",
         {"Italian": r"italian|italia|trattoria|osteria|pizz|pasta", "a time": r"\b[678][:.]\d{2}\s?(pm|p\.m\.)|\b(18|19|20)[:.]\d{2}\b", "Manchester": r"manchester|\bM[1-4]\b",
          "four people": r"\b(4|four)\s?(people|guests|diners|covers|persons|adults)|(party|table) (of|for) (4|four)"},
         links=1, stops=True, long=True, suite="everyday",
         acted=r"\b[678][:.]\d{2}|\b(18|19|20)[:.]\d{2}|reserve|book|select|find a table|continue"),
    Real("gymcancel", "subscription", "how to cancel a gym membership and what is still charged, from the gym's own site",
         "I want to cancel my Planet Fitness membership. What exactly do I have to do, and will I still get charged "
         "anything after? Check their official site.",
         {"the steps": STEPS, "how to cancel": r"in person|in-person|letter|certified mail|home club|online|the app",
          "what is still charged": r"annual fee|\bfee\b|charge|billing|\$\d"}, links=1, suite="everyday"),
    Real("dinners", "meals", "five dinners planned, turned into one shopping list with quantities, and priced at a supermarket",
         "Plan five vegetarian dinners for two for next week, turn them into one shopping list with quantities, and "
         "price it all at Tesco (or another UK supermarket site if Tesco won't load). What's the total?",
         {"five dinners": r"(mon|day 1)[\s\S]*(tue|day 2)[\s\S]*(wed|day 3)[\s\S]*(thu|day 4)[\s\S]*(fri|day 5)|"
                          r"1[.)][\s\S]*2[.)][\s\S]*3[.)][\s\S]*4[.)][\s\S]*5[.)]",
          "quantities": r"(\b\d+(\.\d+)?\s?(g|kg|ml|l|x|packs?|tins?|cans?|bunch(es)?|bags?|jars?|bottles?)\b[\s\S]*?){4}",
          "a total": r"total"},
         links=2, prices=6, tool="run_python", needs=("browser", "sandbox"), long=True, suite="everyday"),
    Real("laufplan", "plan", "a beginner's eight-week 10 km plan as a table, answered in German",
         "Ich bin noch nie mehr als 3 km am Stück gelaufen. Mach mir einen 8-Wochen-Plan für meinen ersten "
         "10-km-Lauf, Start am Montag, 4. Januar 2027, als Tabelle mit den Läufen jeder Woche.",
         {"German": GERMAN, "a table": r"\|.*\|", "eight weeks": r"woche\s*8\b|\|\s*8\s*\|", "rest days": r"ruhetag|\bruhe\b|pause|\bfrei\b|erholung",
          "the 10 km": r"10[\s-]?km|10\s?k\b", "the dates": r"4\.\s?(1\.|jan)|04\.01\.|januar|2027"},
         links=0, needs=(), suite="everyday"),
    Real("jobs", "jobs", "three data-analyst jobs in Berlin for 1-3 years' experience with SQL and Python, and one "
         "application taken to the page that wants a CV",
         "Find me three data analyst jobs in Berlin, on-site or hybrid, for someone with 2 years of experience who "
         "works in SQL, Python and Tableau and speaks English but little German. Only roles posted in the last two "
         "weeks that ask for 0-3 years. For each, give the company, the experience they ask for, the salary if they "
         "show it, and the link. Then start applying to the one that fits best and stop when it asks for my CV or "
         "personal details.",
         {"data analyst": r"data analyst|datenanalyst|data analytics", "Berlin": r"berlin", "SQL": r"\bsql\b",
          "the experience asked": r"\b\d\s*(?:[-–]|to)?\s*\d?\s*\+?\s*(?:years?|yrs?|jahre)|\bentry[- ]level\b|"
                                   r"berufserfahrung|no experience"},
         links=3, stops=True, long=True, suite="everyday", acted=r"apply|bewerb|application|continue|weiter"),
    Real("konzert", "events", "two Tokyo concerts in early 2027 with tickets under a budget, answered in Japanese",
         "2027年2月か3月の土日に東京23区内であるクラシック（オーケストラ）のコンサートで、チケットが1枚1万円以下のものを"
         "2つ探してください。売り切れのものは除いて、公演名・日時・会場・料金・リンクを教えてください。",
         {"Japanese": r"です|ます|ません|ました|(?:日時|公演日|会場|料金|開演|チケット).+?(?:日時|公演日|会場|料金|開演|チケット)",
          "an orchestra": r"オーケストラ|交響|フィル|管弦楽|クラシック|symphon|philharmon",
          "dates": r"(?<!\d)[23]月|\b2027[/.-]0?[23]\b|\b0?[23]/\d{1,2}\b|"
                   r"\b(?:\d{1,2}\s+(?:February|March)|(?:February|March)\s+\d{1,2})\b",
          "a venue": r"ホール|アリーナ|ドーム|武道館|劇場|会館|スタジアム|センター|ガーデン|ライブハウス|クラブ|zepp|hall|arena"},
         links=1, prices=2, suite="everyday"),
    Real("monitor", "shopping", "three 4K monitors for photo editing under a budget, compared in a table with a pick",
         "I need a 27-inch 4K monitor for photo editing, around $400 max. Compare three from Best Buy or Amazon on "
         "color accuracy, ports and price in a table, and tell me which one to get.",
         {"a table": r"\|.*\|", "4K": r"4K|3840", "colour accuracy": r"sRGB|DCI|adobe rgb|delta\s?e|ΔE|colou?r (accura|gamut|cover)",
          "ports": r"HDMI|DisplayPort|USB-?C|\bDP\b", "a recommendation": CHOICE}, links=2, prices=3, suite="everyday"),
    Real("onibus", "travel", "three morning buses from São Paulo to Rio on a date, answered in Portuguese",
         "Quero ir de ônibus de São Paulo para o Rio de Janeiro na sexta-feira, 5 de fevereiro de 2027, de manhã. Me "
         "passa três opções com horário de saída e chegada, empresa, preço e link.",
         {"Portuguese": PORTUGUESE, "times": TIMES,
          "a bus company": r"cometa|1001|expresso|[úu]til\b|kaissara|[áa]guia branca|catarinense|sampaio|penha|itapemirim|via[çc][ãa]o"},
         links=1, prices=3, suite="everyday"),
    Real("recharge", "bills", "the cheapest prepaid plan of two Indian carriers for a data need, compared",
         "Which Jio and Airtel prepaid plans give at least 2 GB a day for 84 days? Put the cheapest one from each in a "
         "table with price, data, calls and the link, and tell me which is better value.",
         {"a table": r"\|.*\|", "Jio": r"\bjio\b", "Airtel": r"airtel", "84 days": r"84\s?(-\s?)?(days?|d\b)",
          "2 GB a day": r"(2|2\.5|3)\s?GB\s?(/|per|a|each)\s?day|(2|2\.5|3)\s?GB/d",
          "a verdict": r"better value|best value|cheaper|go with|recommend|pick"}, links=2, prices=2, suite="everyday"),
    Real("subscriptions", "finance", "the monthly subscriptions in an attached bank statement, totalled in the sandbox",
         "Here's my bank statement for the last three months. Which subscriptions am I paying every month, and what "
         "do they come to per month and per year?",
         # Netflix 10.99, Spotify 11.99, PureGym 24.99 and iCloud 2.99 each month; Audible (once) and Prime (yearly) are not.
         {"Netflix": r"netflix", "Spotify": r"spotify", "PureGym": r"pure\s?gym", "iCloud": r"icloud|apple",
          "the monthly total": r"50\.96", "the yearly total": r"611\.52"},
         links=0, files=("statement.csv",), tool="run_python", needs=("sandbox",), suite="everyday"),
    Real("brief", "routine", "a weekday morning brief set up as a routine at the time asked",
         "Every weekday at 7:30 in the morning, give me the weather in Mumbai for the day and the top three headlines "
         "from The Hindu.",
         {"the routine": r"7[:.]30|weekday|routine|every (morning|weekday)"}, links=0, needs=("routines",), suite="everyday",
         routine=r"weekdays 07:30", routine_does=(r"weather|forecast|temperature", r"hindu")),
    Real("preiswecker", "routine", "a daily price watch set up as a routine, with today's price, answered in German",
         "Behalte den Preis vom Sony WH-1000XM6 auf amazon.de im Auge: schau jeden Morgen um 9 Uhr nach und sag mir "
         "Bescheid, sobald er unter 300 € fällt. Was kostet er gerade?",
         {"German": GERMAN, "the headphones": r"WH-?\s?1000\s?XM6"}, links=1, prices=1,
         needs=("browser", "routines"), suite="everyday", routine=r"daily 09:00", routine_does=(r"1000\s?-?XM6", r"300")),
    Real("passportfees", "routine", "a weekly watch on a government fees page set up as a routine, with today's fee",
         "Keep an eye on the GOV.UK passport fees page for me: check it every Monday at 9am and let me know if the fee "
         "for an adult passport changes. What is it right now?",
         {"the fee": r"£\s?\d{2,3}(\.\d{2})?", "adult": r"adult|16 or over|over 16|aged 16"}, links=1, prices=1,
         needs=("browser", "routines"), suite="everyday", routine=r"weekly mon 09:00",
         routine_does=(r"passport", r"fee|price|cost|£")),
]
