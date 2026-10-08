from datetime import datetime

import pytest

from gemma_memory.dates import anchor

TUESDAY = datetime(2023, 5, 23, 10, 0).timestamp()
SUNDAY = datetime(2023, 5, 21, 10, 0).timestamp()


@pytest.mark.parametrize("text, expected", [
    ("I finally beat it last weekend!", "I finally beat it last weekend [2023-05-20/21]!"),
    ("We met yesterday.", "We met yesterday [2023-05-22]."),
    ("Back last Friday it rained.", "Back last Friday [2023-05-19] it rained."),
    ("On Saturday we hiked.", "On Saturday [2023-05-20] we hiked."),
    ("I started 3 weeks ago.", "I started 3 weeks ago [≈ 2023-05-02]."),
    ("my sister came two days ago", "my sister came two days ago [≈ 2023-05-21]"),
    ("Last month I moved.", "Last month [2023-04] I moved."),
    ("a few months ago", "a few months ago [≈ 2023-02]"),
    ("last year too", "last year [2022] too"),
    ("this weekend is free", "this weekend [2023-05-27/28] is free"),
])
def test_relative_expressions_get_their_date(text, expected):
    assert anchor(text, TUESDAY) == expected


@pytest.mark.parametrize("text", [
    "Next weekend I fly out.",  # the coming weekend or the one after: left alone rather than guessed
    "next Friday works",
    "Back in 2019 I lived there.",
    "no time words here",
])
def test_ambiguous_or_absolute_text_is_left_alone(text):
    assert anchor(text, TUESDAY) == text


def test_last_weekend_said_on_a_weekend_is_the_one_before():
    assert anchor("last weekend", SUNDAY) == "last weekend [2023-05-13/14]"


MARCH_17 = datetime(2023, 3, 17, 18, 54).timestamp()


@pytest.mark.parametrize("text, expected", [
    ("twins, who were born on February 12th", "twins, who were born on February 12th [2023-02-12]"),
    ("the party is on the 3rd of April", "the party is on the 3rd of April [2023-04-03]"),
    ("we met on Dec 28", "we met on Dec 28 [2022-12-28]"),  # nearest occurrence: last December
    ("since Feb 3", "since Feb 3 [2023-02-03]"),
    ("May 5th was fun", "May 5th [2023-05-05] was fun"),
])
def test_month_day_dates_get_the_nearest_year(text, expected):
    assert anchor(text, MARCH_17) == expected


@pytest.mark.parametrize("text", ["signed February 12, 2023 already", "in May 5 people came", "may I ask"])
def test_dated_or_ambiguous_month_words_are_left_alone(text):
    assert anchor(text, MARCH_17) == text
