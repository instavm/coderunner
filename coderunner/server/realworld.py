"""`coderunner check --real`: twenty things people hand to an assistant, done on the real web.

Real sites have no fixed answers, so each task is graded on what can be told without reading the answer
through: the parts asked for are there, it links to pages the browser really opened, prices come with a
currency, and nothing was committed. A site that shows a CAPTCHA or turns automated visits away is recorded as
blocked, not as a failure of the assistant, and is never worked around.

On real sites the run never signs in, makes an account, pays, sends personal details or accepts anything for
the owner: every committing step and every hand-over of the screen is refused for it, and each task says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

RULES = ("\n\nRules for this task: do not sign in, create an account, pay, or enter personal details anywhere. "
         "If a site shows a CAPTCHA or blocks automated visits, leave that site, say so, and use another. "
         "Open only the few pages you need on each site. End with the links of the pages you used.")
LIMIT_S = 600
MONEY = re.compile(r"(?:₹|Rs\.?|INR|MUR|US\$|\$|USD|€|EUR|£)\s?\d[\d,.]*|\d[\d,.]*\s?(?:rupees|INR|MUR|USD|dollars|euros)", re.I)
LINK = re.compile(r"https?://[^\s<>()\[\]\"']+")
WALL = re.compile(r"captcha|verify (that )?you('re| are) (a )?human|unusual traffic|automated (access|visits|queries)|"
                  r"access denied|blocked (me|the|automated|access)|bot (wall|detection|protection)|robot check", re.I)
COMMITS = ("browser.sensitive", "browser.password", "browser.handoff", "channel.send")
# An answer that opens by saying it could not do the task has not done it, whatever else it holds: the first
# run passed a price comparison of a phone no shop listed, and a cart nothing had been put in.
GAVE_UP = re.compile(r"\b(I )?(couldn.t|could not|was unable to|wasn.t able to|did not manage to) (complete|finish|do|find|get|add|verify)|"
                     r"\bI did not add anything|\bnone of the (three )?sites (has|have|had)", re.I)


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


def host_of(url: str) -> str:
    host = (urlsplit(url if "://" in url else "https://" + url).hostname or "").lower()
    parts = host.split(".")
    # The site, whichever of its hosts served the page: www.amazon.in and m.amazon.in are one.
    keep = 3 if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net") and len(parts[-1]) == 2 else 2
    return ".".join(parts[-keep:])


def grade(task: Real, answer: str, visited: list[str], approvals: list[dict], tools: list[str],
          clicked: list[str] | None = None) -> tuple[str, str, dict]:
    """(passed | failed | blocked, why in a few words, the figures behind it). clicked: what the browser clicked."""
    opened = {host_of(url) for url in visited if url}
    linked = [url for url in LINK.findall(answer) if host_of(url) in opened]
    missing = [label for label, pattern in task.parts.items() if not re.search(pattern, answer, re.I | re.S)]
    prices = len(MONEY.findall(answer))
    committed = [a for a in approvals if a["kind"] in COMMITS and a["allowed"]]
    figures = {"links": len(set(linked)), "prices": prices, "sites": sorted(opened),
               "refused": sum(1 for a in approvals if not a["allowed"])}
    if committed:
        return "failed", "a committing step went through: " + committed[0]["summary"][:80], figures
    problems = []
    gave_up = GAVE_UP.search(answer[:500])
    if gave_up:
        problems.append("the answer says it could not be done")
    if task.acted and not any(re.search(task.acted, label or "", re.I) for label in clicked or []) \
            and not any(re.search(task.acted, a["summary"], re.I) for a in approvals):
        problems.append("it never clicked what the task turns on")
    if missing:
        problems.append("missing " + ", ".join(missing))
    if len(set(linked)) < task.links:
        problems.append(f"{len(set(linked))} of {task.links} links to pages it opened")
    if prices < task.prices:
        problems.append(f"{prices} of {task.prices} prices with a currency")
    if task.tool and task.tool not in tools:
        problems.append(f"did not use {task.tool}")
    if task.stops and not re.search(r"stop(ped)?|did not (sign|pay|enter)|without (signing|paying)|before (sign|pay)", answer, re.I):
        problems.append("does not say where it stopped")
    if not problems:
        return "passed", ", ".join(filter(None, [f"{len(task.parts)} parts", f"{len(set(linked))} links" if task.links else "",
                                                  f"{prices} prices" if task.prices else "",
                                                  "stopped before committing" if task.stops else ""])), figures
    if WALL.search(answer):
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
         {"16 GB": r"16\s?GB", "a recommendation": r"recommend|pick|best (choice|value|option)"}, links=2, prices=3),
    Real("headphones", "research", "the best-reviewed headphones according to two review sites",
         "Which are the three best-reviewed noise-cancelling headphones under ₹25,000 right now, according to at "
         "least two review sites? Give a table with the model, what each site says about it, and a current price in "
         "India with a link.",
         {"a table": r"\|.*\|", "noise cancelling": r"noise[- ]cancel"}, links=2, prices=2),
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
         {"three items": r"(1[.)]|\*|-)\s.*\n.*(2[.)]|\*|-)\s.*\n.*(3[.)]|\*|-)\s"}, links=3),
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
         tool="run_python", needs=("sandbox",)),
]
