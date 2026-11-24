"""Which link clicks only open another page, so the step judge need not be asked.

On the owner's model server each judgement made the task read its whole conversation again (~11,800 tokens,
23 s): a request in the judge's slot throws away what the task's slot holds (safety/decision.py, ActionCheck).
9 research tasks asked the judge 183 times, mostly for clicks, and many of those were links to another page.
A click on such a link does what opening its address does, which needs no judge. Only a plain one is let
through: an <a> with a web address that leads off the current page, where no commit word appears in its
name, the words around it, its address or the address of the page it is on. Everything else (buttons,
fields, dropdowns, typing, forms, payment fields, links that run a script) is judged as before. The list
errs towards asking: a word in it only sends the click to the judge.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.parse

# Words of committing steps in English, German, French, Spanish, Portuguese, Hindi, Japanese and Chinese.
# Latin-script entries match whole words (\w* where inflections differ); joined forms are listed as well, for
# addresses (add-to-cart and addToCart both read as words here, addtocart as one).
LATIN = [
    # English
    r"buy\w*", r"pay", r"pays", r"paying", r"payments?", r"pay ?now", r"order", r"ordering", r"place ?order",
    r"place (?:an|the|your|my) (?:order|bid)", r"check ?out\w*", r"book", r"booking", r"book ?now",
    r"reserv\w*", r"subscri\w*", r"unsubscri\w*", r"sign ?up", r"enrol\w*", r"regist\w*", r"submit\w*",
    r"send", r"sends", r"sending", r"confirm\w*", r"delet\w*", r"remove", r"(?<!noise )cancel",
    r"(?<!noise )cancel(?:s|led|ling)", r"(?<!free )cancellation", r"apply", r"applying", r"donat\w*",
    r"add(?:ed|ing)? to (?:the |my |your )?(?:cart|basket|bag|trolley|order)",
    r"addto(?:cart|basket|bag|trolley)", r"purchas\w*", r"log ?out", r"log ?off", r"sign ?out", r"follow",
    r"unfollow", r"invite\w*", r"accept\w*", r"agree\w*", r"authori[sz]\w*", r"grant", r"rsvp", r"bid",
    r"post", r"publish\w*", r"share", r"transfer\w*", r"withdraw\w*", r"deposit\w*", r"redeem\w*",
    r"upgrade\w*", r"renew\w*", r"join", r"vote", r"deactivat\w*", r"unlink",
    # German
    r"kauf\w*", r"bezahl\w*", r"zahlung\w*", r"bestell\w*", r"zur kasse", r"kasse", r"buchen", r"buchung\w*",
    r"reservier\w*", r"abonnier\w*", r"abbestell\w*", r"abmeld\w*", r"anmeld\w*", r"registrier\w*",
    r"absend\w*", r"abschick\w*", r"senden", r"bestätig\w*", r"lösch\w*", r"entfern\w*", r"stornier\w*",
    r"kündig\w*", r"bewerb\w*", r"spende\w*", r"spenden", r"in den (?:warenkorb|einkaufswagen)", r"ausloggen",
    # French
    r"achet\w*", r"achat\w*", r"payer", r"paiement\w*", r"command\w*", r"réserv\w*", r"abonn\w*",
    r"désabonn\w*", r"inscri\w*", r"envoy\w*", r"soumettre", r"valider", r"supprim\w*", r"annul\w*",
    r"postul\w*", r"faire un don", r"ajouter au (?:panier|caddie)", r"déconnect\w*", r"déconnexion",
    # Spanish
    r"comprar", r"compra", r"cómpralo", r"pagar", r"pago", r"pedido\w*", r"suscrib\w*", r"darse de baja",
    r"date de baja", r"regístr\w*", r"enviar", r"envía\w*", r"elimin\w*", r"borrar", r"solicit\w*", r"donar",
    r"donaci\w*", r"(?:añadir|agregar|añade|agrega) al (?:carrito|carro|cesta)", r"cerrar sesi\w*",
    # Portuguese
    r"compre", r"pagamento\w*", r"finalizar (?:compra|pedido)", r"fazer (?:o )?pedido", r"assin\w*",
    r"inscrev\w*", r"cadastr\w*", r"envie", r"excluir", r"apagar", r"candidat\w*", r"doar", r"doe",
    r"adicionar (?:ao|à) (?:carrinho|sacola|cesta)", r"sair", r"encerrar sess\w*", r"desconectar",
]
LATIN_WORDS = re.compile(r"\b(?:" + "|".join(LATIN) + r")\b", re.I)
# Hindi: a word's start (inflections follow it), not inside another word ("दान", donate, is in "प्रदान").
HINDI = ["खरीद", "भुगतान", "ऑर्डर", "आर्डर", "बुक", "आरक्षण", "सदस्यता", "रजिस्टर", "पंजीकरण", "पंजीकृत", "जमा",
         "भेज", "पुष्टि", "हटा", "मिटा", "रद्द", "आवेदन", "दान", "कार्ट में", "लॉग आउट", "साइन अप",
         "सब्सक्राइब", "चेकआउट"]
HINDI_WORDS = re.compile(r"(?<![ऀ-ॿ])(?:" + "|".join(HINDI) + ")")
# Japanese and Chinese have no spaces between words: anywhere in the text.
CJK = ["購入", "買う", "カートに入れる", "カートに追加", "かごに入れる", "支払", "決済", "注文", "予約", "申し込",
       "申込", "登録", "送信", "確定", "削除", "キャンセル", "取り消", "取消", "解約", "退会", "寄付", "応募",
       "ログアウト", "購読", "配信停止", "レジに進む", "投稿", "同意",
       "购买", "支付", "付款", "下单", "预订", "预定", "订阅", "注册", "提交", "发送", "确认", "删除", "取消",
       "加入购物车", "捐款", "退出登录"]
CJK_WORDS = re.compile("|".join(CJK))

# A parameter whose value is a date, a number or a sort order says nothing by its name: booking.com's hotel
# links carry checkout=2026-12-03 and order=popularity, which open a page and commit nothing. A lone 0 or 1 is
# a switch, and its name says what it does (addToCart=1).
QUIET_VALUE = re.compile(r"(?![01]$)[\d\s\-/:.T+]+|(?:asc|desc)\w*|price\w*|popular\w*|rating\w*|relevan\w*|newest|"
                         r"date\w*|distance|name|review\w*", re.I)
LONGEST_LINK = 300          # the page script cuts a link's address here: a cut address may hide its words


def commit_word(text: str) -> bool:
    """Whether text holds a word of a committing step in any of the languages above."""
    if not text:
        return False
    text = unicodedata.normalize("NFKC", text)
    return bool(LATIN_WORDS.search(text) or HINDI_WORDS.search(text) or CJK_WORDS.search(text))


def address_words(url: str) -> str:
    """The words of an address's path and query, separated as words are (add-to-cart, add_to_cart and
    addToCart all read "add to cart", or addtocart), the names of quiet parameters left out."""
    parts = urllib.parse.urlsplit(url)
    words = [urllib.parse.unquote_plus(parts.path)]
    for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True):
        words.append(value)
        if not QUIET_VALUE.fullmatch(value.strip()):
            words.append(key)
    # Devanagari's vowel signs are not word characters to re, so their block is kept as a whole.
    text = unicodedata.normalize("NFKC", " ".join(words))
    return re.sub(r"(?:[^\w\u0900-\u097F]|_)+", " ", text).strip()


def plain_link(tag: str, role: str, href: str, page_url: str, *texts: str) -> bool:
    """A click that only opens another web page: an <a> the page presents as a link, whose address is a web
    page other than this one (an address that differs only after # is a script's control or a jump down the
    page), short enough to be read whole, with no commit word in its words, its address or this page's."""
    if tag != "a" or role != "link" or not href or len(href) >= LONGEST_LINK:
        return False
    link = urllib.parse.urlsplit(href)
    if link.scheme not in ("http", "https") or not link.hostname:
        return False
    if href.split("#")[0] == page_url.split("#")[0]:
        return False
    return not any(commit_word(text) for text in (*texts, address_words(href), address_words(page_url)))
