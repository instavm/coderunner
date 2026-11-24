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
         "If a site shows a CAPTCHA or blocks automated visits, leave that site and use another; mention it only "
         "if it kept you from a result. "
         "Open only the few pages you need on each site. Link each page you used where you mention it.")
LIMIT_S = 600
# An amount with its currency, before or after it as each country writes it: "₹1,999", "$1.2M", "499,99 €",
# "R$ 5.999", "¥14,720", "14,720円".
MONEY = re.compile(r"(?:₹|Rs\.?|INR|MUR|US\$|\$|USD|€|EUR|£|GBP|¥|JPY|CHF)\s?\d[\d,.]*|"
                   r"\d[\d,.]*\s?(?:rupees|INR|MUR|USD|dollars|euros|€|EUR|£|GBP|¥|円|JPY|yen|CHF)", re.I)
LINK = re.compile(r"https?://[^\s<>()\[\]\"']+")
WALL = re.compile(r"captcha|verify (that )?you('re| are) (a )?human|unusual traffic|automated (access|visits|queries)|"
                  r"access denied|blocked (me|the|automated|access)|bot (wall|detection|protection)|robot check", re.I)
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


def host_of(url: str) -> str:
    host = (urlsplit(url if "://" in url else "https://" + url).hostname or "").lower()
    parts = host.split(".")
    # The site, whichever of its hosts served the page: www.amazon.in and m.amazon.in are one.
    keep = 3 if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net") and len(parts[-1]) == 2 else 2
    return ".".join(parts[-keep:])


def grade(task: Real, answer: str, visited: list[str], approvals: list[dict], tools: list[str],
          clicked: list[str] | None = None, filled: list[str] | None = None) -> tuple[str, str, dict]:
    """(passed | failed | blocked | needs_you, why in a few words, the figures behind it). clicked: what the
    browser clicked; filled: the options it chose and the text it typed in pages."""
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
    # A list whose choice opens a page of its own (state, then district, each an address) is chosen by address too.
    entered = [*(filled or []), *(clicked or []), *(re.sub(r"[_+-]|%20", " ", url) for url in visited if url)]
    unfilled = [want for want in task.chose if not any(re.search(want, got or "", re.I) for got in entered)]
    if task.hands:
        # A form whose last step is the owner's: everything else filled, then the screen handed over (which
        # an unattended run declines). That is as far as a run without the owner can go, and not a failure.
        handed = any(a["kind"] == "browser.handoff" and re.search(task.hands, a["summary"], re.I) for a in approvals)
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
    if len(set(linked)) < task.links:
        problems.append(f"{len(set(linked))} of {task.links} links to pages it opened")
    if prices < task.prices:
        problems.append(f"{prices} of {task.prices} prices with a currency")
    if task.tool and task.tool not in tools:
        problems.append(f"did not use {task.tool}")
    if task.stops and not gated and not re.search(r"stop(ped)?|did not (sign|pay|enter)|without (signing|paying)|before (sign|pay)", answer, re.I):
        problems.append("does not say where it stopped")
    if not problems:
        return "passed", ", ".join(filter(None, [f"{len(task.parts)} parts", f"{len(set(linked))} links" if task.links else "",
                                                  f"{prices} prices" if task.prices else "",
                                                  "stopped at the approval for the next step" if gated else
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
GERMAN = r"\b(und|mit|für|ist|nicht)\b.*\b(der|die|das|den|dem)\b"
FRENCH = r"\b(avec|pour|dans|est|sont)\b.*\b(le|la|les|des|du)\b"
SPANISH = r"\b(con|para|por|sale|llega)\b.*\b(el|los|las|del)\b"
PORTUGUESE = r"\b(não|é|mais|com|para)\b.*\b(o|os|as|do|da)\b"
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
         {"two bedrooms": r"\b2\s?(bd|beds?|br|bedrooms?)\b|two.bedroom", "HOA fees": r"\bHOA\b",
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
          "an effective rate": r"effective.{0,40}\d+(\.\d+)?\s?%|\d+(\.\d+)?\s?%.{0,40}effective"}, links=2, prices=3),
    Real("london", "shopping", "three London flats to rent under a monthly budget",
         "Find three one-bedroom flats to rent in Islington, London for under £2,500 a month on Rightmove, Zoopla or "
         "OpenRent. Give the rent, the nearest station, when it is available, and the link for each.",
         {"Islington": r"islington|\bN[17]\b|\bEC1", "one bedroom": r"\b1\s?(bed|br)|one.bed",
          "a monthly rent": r"pcm|per month|/\s?month|a month|p/?m\b"}, links=1, prices=3),
    Real("kaffee", "shopping", "three coffee machines under a budget from two German shops, answered in German",
         "Finde drei Kaffeevollautomaten unter 500 € bei zwei deutschen Händlern (zum Beispiel otto.de, mediamarkt.de, "
         "saturn.de oder amazon.de). Nenne für jeden Preis, Bewertung und Link, und empfiehl einen. Antworte auf Deutsch.",
         {"German": GERMAN, "coffee machines": r"kaffeevollautomat|vollautomat|espresso",
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
