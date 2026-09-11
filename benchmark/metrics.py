"""Corpus error rates: summed edit distance / summed reference units."""

import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def counts(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    words, chars = reference.split(), reference.replace(" ", "")
    return {
        "word_errors": distance(words, hypothesis.split()),
        "reference_words": len(words),
        "char_errors": distance(chars, hypothesis.replace(" ", "")),
        "reference_chars": len(chars),
    }


def rates(totals):
    return {
        **totals,
        "wer": totals["word_errors"] / totals["reference_words"]
        if totals["reference_words"]
        else None,
        "cer": totals["char_errors"] / totals["reference_chars"]
        if totals["reference_chars"]
        else None,
    }
