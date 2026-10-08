"""Relative time expressions anchored to the date the message was written, for the injected text only.

"I finally beat it last weekend" written on 2023-05-23 becomes "... last weekend [≈ 2023-05-20/21]": the reader no
longer has to do calendar arithmetic from the conversation header, which is where temporal questions go wrong. The
original words are kept; an expression that can't be resolved is left as it is. Rule-based, English, no model.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "a couple of": 2, "a few": 3, "couple of": 2, "few": 3}
_UNIT_DAYS = {"day": 1, "week": 7}

_PATTERNS = [
    re.compile(r"\b(today|tonight|this morning|this afternoon|this evening)\b", re.I),
    re.compile(r"\b(yesterday|last night)\b", re.I),
    re.compile(r"\b(the day before yesterday)\b", re.I),
    re.compile(r"\b(tomorrow)\b", re.I),
    re.compile(r"\b(last|this past|past) (weekend)\b", re.I),
    re.compile(r"\b(this) (weekend)\b", re.I),  # "next weekend" is ambiguous (the coming one or the one after)
    re.compile(r"\b(last|this|next) (week|month|year)\b", re.I),
    re.compile(r"\b(last|on|this past) (" + "|".join(_WEEKDAYS) + r")\b", re.I),  # "next Friday": ambiguous too
    re.compile(r"\b(\d+|" + "|".join(sorted(_NUM, key=len, reverse=True)) + r") (day|week|month|year)s? ago\b", re.I),
]


def _iso(d: date) -> str:
    return d.isoformat()


def _resolve(m: re.Match, today: date) -> str | None:
    g = [x.lower() if x else x for x in m.groups()]
    text = m.group(0).lower()
    if text in ("today", "tonight", "this morning", "this afternoon", "this evening"):
        return _iso(today)
    if text in ("yesterday", "last night"):
        return _iso(today - timedelta(days=1))
    if text == "the day before yesterday":
        return _iso(today - timedelta(days=2))
    if text == "tomorrow":
        return _iso(today + timedelta(days=1))
    if text.endswith("ago"):
        n = int(g[0]) if g[0].isdigit() else _NUM.get(g[0])
        if not n or n > 400:
            return None
        if g[1] in _UNIT_DAYS:
            return "≈ " + _iso(today - timedelta(days=n * _UNIT_DAYS[g[1]]))
        if g[1] == "month":
            y, mo = today.year, today.month - n
            y, mo = y + (mo - 1) // 12, (mo - 1) % 12 + 1
            return f"≈ {y}-{mo:02d}"
        return f"≈ {today.year - n}"
    if len(g) == 2 and g[1] == "weekend":
        sat = today - timedelta(days=(today.weekday() - 5) % 7)  # most recent Saturday (today if Saturday)
        if g[0] in ("last", "this past", "past"):
            if today.weekday() >= 5:  # said on the weekend itself: "last weekend" is the one before
                sat -= timedelta(days=7)
        else:  # this / next weekend
            sat = today + timedelta(days=(5 - today.weekday()) % 7)
            if g[0] == "next":
                sat += timedelta(days=7)
        return f"{_iso(sat)}/{(sat + timedelta(days=1)).day:02d}"
    if len(g) == 2 and g[1] in ("week", "month", "year"):
        shift = {"last": -1, "this": 0, "next": 1}[g[0]]
        if g[1] == "week":
            monday = today - timedelta(days=today.weekday()) + timedelta(weeks=shift)
            return f"week of {_iso(monday)}"
        if g[1] == "month":
            y, mo = today.year, today.month + shift
            y, mo = y + (mo - 1) // 12, (mo - 1) % 12 + 1
            return f"{y}-{mo:02d}"
        return str(today.year + shift)
    if len(g) == 2 and g[1] in _WEEKDAYS:
        target = _WEEKDAYS.index(g[1])
        if g[0] == "next":
            delta = (target - today.weekday()) % 7 or 7
            return _iso(today + timedelta(days=delta))
        back = (today.weekday() - target) % 7 or 7  # last/on/this past: the most recent such day before today
        return _iso(today - timedelta(days=back))
    return None


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
# "February 12th" / "Feb 12" / "12th of February" / "12 February", but not when a year follows ("February 12, 2023").
_DATE_WORDS = {"on", "since", "until", "till", "by", "before", "after", "from", "of", "around"}
_MONTH_DAY = [
    re.compile(_MON + r"\.? (\d{1,2})(?:st|nd|rd|th)?\b(?!,? ?\d{4})", re.I),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)? (?:of )?" + _MON + r"\b(?!,? ?\d{4})", re.I),
]


def _month_day(m: re.Match, today: date) -> str | None:
    a, b = m.groups()
    month_word, day = (a, b) if not a.isdigit() else (b, a)
    month = next((i + 1 for i, name in enumerate(_MONTHS) if name.startswith(month_word.lower()[:3])), None)
    if month == 5 and not a.isdigit():  # "May 5" is often not a date ("in May 5 people came"): need more evidence
        before = m.string[:m.start()].split()[-1:]
        if not re.search(r"\d(st|nd|rd|th)\b", m.group(0), re.I) and (before or [""])[0].lower() not in _DATE_WORDS:
            return None
    day = int(day)
    if not month or not 1 <= day <= 31:
        return None
    candidates = []
    for year in (today.year - 1, today.year, today.year + 1):  # the occurrence nearest to when it was written
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            pass
    return _iso(min(candidates, key=lambda d: abs((d - today).days))) if candidates else None


def anchor(text: str, written_at: float) -> str:
    """``text`` with each resolvable relative expression, and each month-day date without a year, followed by its
    date in brackets."""
    today = datetime.fromtimestamp(written_at).date()
    spans = []
    for pat in _MONTH_DAY:
        for m in pat.finditer(text):
            if any(s <= m.start() < e or s < m.end() <= e for s, e, _ in spans):
                continue
            resolved = _month_day(m, today)
            if resolved:
                spans.append((m.start(), m.end(), resolved))
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            if any(s <= m.start() < e or s < m.end() <= e for s, e, _ in spans):
                continue
            resolved = _resolve(m, today)
            if resolved:
                spans.append((m.start(), m.end(), resolved))
    for start, end, resolved in sorted(spans, reverse=True):
        text = text[:end] + f" [{resolved}]" + text[end:]
    return text
