import re, html, unicodedata, emoji
from bs4 import BeautifulSoup

HTML_RE      = re.compile(r"<[^>]+>")
BRAND_RE     = re.compile(r"\b(?:®|™|\(.*?officiel\))\b", flags=re.I)
WHITESPACE_RE= re.compile(r"\s+")

def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(" ")

def strip_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace='')

def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = strip_html(text)
    text = strip_emojis(text)
    text = BRAND_RE.sub(" ", text)
    text = normalise_unicode(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()
