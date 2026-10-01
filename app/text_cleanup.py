"""Repairs to text extracted from newspaper e-paper PDFs.

Typeset editions (e.g. The Hindu's e-paper) come out of pypdf with
ligatures as Unicode presentation forms (ﬁ, ﬀ) or as glyph names like
`/f_i`, and with words hyphenated across column line breaks ("De-\npartment").
"""
from __future__ import annotations

import re
import unicodedata

_GLYPH_LIGATURES = [("/f_f_i", "ffi"), ("/f_f_l", "ffl"), ("/f_f", "ff"), ("/f_i", "fi"), ("/f_l", "fl")]
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")


def clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ﬁ -> fi, ﬀ -> ff, ...
    for glyph, letters in _GLYPH_LIGATURES:
        text = text.replace(glyph, letters)
    return _HYPHEN_BREAK.sub(r"\1\2", text)
