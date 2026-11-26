"""
humanizer.py - Humanization STYLE editor

Rewrites stiff, template-like prose so it reads more natural: swaps
stock "AI-sounding" phrases, adds contractions, varies sentence rhythm
and trims over-used transitions. It is a style tool: always re-read the
output and make sure it still says what YOU mean and is your own work.
"""
import random
import re

# regex -> list of alternatives ("" removes the phrase)
EN_REPLACEMENTS = [
    (r"\bIn conclusion,?\s*", ["So, ", "All in all, ", "To wrap up, "]),
    (r"\bIn summary,?\s*", ["Basically, ", "In short, "]),
    (r"\bTo summarize,?\s*", ["In short, ", "Put simply, "]),
    (r"\bMoreover,?\s*", ["Also, ", "On top of that, ", "Plus, "]),
    (r"\bFurthermore,?\s*", ["Also, ", "What's more, ", ""]),
    (r"\bAdditionally,?\s*", ["Also, ", "Besides, ", ""]),
    (r"\bUltimately,?\s*", ["In the end, ", ""]),
    (r"\bIt is important to note that\s*", ["Note that ", "Keep in mind that ", ""]),
    (r"\bIt is worth noting that\s*", ["Notably, ", "Worth knowing: ", ""]),
    (r"\bIt's worth noting that\s*", ["Notably, ", ""]),
    (r"\bplays? a crucial role in\b", ["matters a lot for", "is key to", "really helps with"]),
    (r"\bplays? a vital role in\b", ["is important for", "is central to"]),
    (r"\bIn today's (?:fast-paced |modern )?world,?\s*", ["Today, ", "These days, ", "Nowadays, "]),
    (r"\ba myriad of\b", ["lots of", "many", "plenty of"]),
    (r"\ba wide range of\b", ["many", "all sorts of", "a lot of"]),
    (r"\bdelve(?:s)? into\b", ["look into", "dig into", "explore"]),
    (r"\bdelving into\b", ["looking into", "digging into"]),
    (r"\bleverag(?:e|es)\b", ["use", "uses"]),
    (r"\bleveraging\b", ["using"]),
    (r"\butili[sz]e\b", ["use"]),
    (r"\bfacilitate\b", ["help with", "make easier"]),
    (r"\bseamless(?:ly)?\b", ["smooth", "smoothly"]),
    (r"\brobust\b", ["strong", "solid"]),
    (r"\bpivotal\b", ["key", "important"]),
    (r"\bparamount\b", ["very important", "top priority"]),
    (r"\bcomprehensive\b", ["complete", "thorough", "full"]),
    (r"\bmultifaceted\b", ["complex", "many-sided"]),
    (r"\bcutting-edge\b", ["modern", "latest"]),
    (r"\bshed(?:s)? light on\b", ["explain", "clear up"]),
    (r"\bin order to\b", ["to"]),
    (r"\bdue to the fact that\b", ["because"]),
    (r"\bin the realm of\b", ["in"]),
    (r"\bnumerous\b", ["many", "plenty of"]),
    (r"\bsubsequently\b", ["then", "later"]),
    (r"\bprior to\b", ["before"]),
    (r"\bendeavou?r(?:s)?\b", ["try", "tries"]),
    (r"\bunderscores?\b", ["shows", "highlights"]),
    (r"\bever-evolving\b", ["changing"]),
]

SW_REPLACEMENTS = [
    (r"\bKwa kumalizia,?\s*", ["Mwisho, ", "Kwa ufupi, ", "Basi, "]),
    (r"\bKwa ujumla,?\s*", ["Kwa jumla, ", "Kwa kifupi, "]),
    (r"\bAidha,?\s*", ["Pia, ", "Vilevile, ", ""]),
    (r"\bIsitoshe,?\s*", ["Zaidi ya hapo, ", "Pia, "]),
    (r"\bZaidi ya hayo,?\s*", ["Pia, ", "Tena, "]),
    (r"\bNi muhimu kutambua kwamba\s*", ["Tukumbuke kuwa ", "Jambo la kukumbuka ni kuwa ", ""]),
    (r"\bNi muhimu kuzingatia kwamba\s*", ["Tuzingatie kuwa ", ""]),
    (r"\bKatika ulimwengu wa leo,?\s*", ["Siku hizi, ", "Leo hii, "]),
    (r"\bina jukumu muhimu katika\b", ["ni muhimu sana katika", "inasaidia sana katika"]),
    (r"\bkwa hakika\b", ["kweli", "hakika"]),
    (r"\bHatimaye,?\s*", ["Mwishowe, ", ""]),
]

CONTRACTIONS = [
    (r"\bdo not\b", "don't"), (r"\bDo not\b", "Don't"),
    (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"),
    (r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"),
    (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"), (r"\bwere not\b", "weren't"),
    (r"\bwill not\b", "won't"), (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"), (r"\bhas not\b", "hasn't"),
    (r"\bhave not\b", "haven't"), (r"\bit is\b", "it's"),
    (r"\bIt is\b", "It's"), (r"\bthat is\b", "that's"),
    (r"\bThat is\b", "That's"), (r"\bthere is\b", "there's"),
    (r"\bThere is\b", "There's"), (r"\bwe are\b", "we're"),
    (r"\bWe are\b", "We're"), (r"\bthey are\b", "they're"),
    (r"\bThey are\b", "They're"), (r"\bI am\b", "I'm"),
    (r"\byou are\b", "you're"), (r"\bYou are\b", "You're"),
    (r"\blet us\b", "let's"), (r"\bLet us\b", "Let's"),
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
SPLIT_POINTS = [", and ", ", but ", "; ", ", which ", ", while ", ", so "]
LEVELS = {"light": 0.35, "medium": 0.65, "strong": 0.9}


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def _word_count(s):
    return len(s.split())


def _replace_phrases(text, rng, lang, p):
    table = SW_REPLACEMENTS if lang == "sw" else EN_REPLACEMENTS

    for pattern, options in table:
        def sub(m, options=options):
            if rng.random() > p + 0.25:
                return m.group(0)
            choice = rng.choice(options)
            original = m.group(0)
            if original[:1].isupper() and choice:
                choice = _cap(choice)
            elif not choice and m.start() == 0:
                return ""
            return choice
        text = re.sub(pattern, sub, text)
    # fix sentence starts left lowercase after phrase removal
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return _cap(text.lstrip())


def _contract(text, rng, p):
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, lambda m, r=repl: r if rng.random() < p + 0.15 else m.group(0), text)
    return text


def _split_long(sentences, rng, p):
    out = []
    for s in sentences:
        if _word_count(s) > 26 and rng.random() < p + 0.2:
            for sp in SPLIT_POINTS:
                idx = s.find(sp)
                if 30 < idx < len(s) - 30:
                    head = s[:idx].rstrip(",;")
                    tail = s[idx + len(sp):]
                    joiner = {"; ": "", ", and ": "", ", so ": "So ",
                              ", but ": "But ", ", which ": "That ", ", while ": "Meanwhile, "}[sp]
                    if joiner == "That ":
                        tail = "That " + tail
                    else:
                        tail = joiner + tail
                    out.append(head + ".")
                    out.append(_cap(tail))
                    break
            else:
                out.append(s)
        else:
            out.append(s)
    return out


def _merge_short(sentences, rng, p):
    out, i = [], 0
    while i < len(sentences):
        s = sentences[i]
        if (i + 1 < len(sentences) and _word_count(s) < 8
                and _word_count(sentences[i + 1]) < 10 and rng.random() < p):
            nxt = sentences[i + 1]
            joiner = rng.choice([", and ", " — ", ", "])
            out.append(s.rstrip(".!?") + joiner + nxt[:1].lower() + nxt[1:])
            i += 2
        else:
            out.append(s)
            i += 1
    return out


def _trim_transition_openers(sentences, rng, p):
    openers = re.compile(r"^(However|Therefore|Thus|Consequently|Hence|Nevertheless|"
                         r"Similarly|Likewise|Hata hivyo|Kwa hiyo|Hivyo basi),\s*", re.I)
    out, last_had = [], False
    for s in sentences:
        has = bool(openers.match(s))
        if has and last_had and rng.random() < p + 0.2:
            s = _cap(openers.sub("", s, count=1))
            has = False
        out.append(s)
        last_had = has
    return out


def _add_touches(sentences, rng, lang):
    """Strong level only: a short rhetorical beat now and then."""
    beats_en = ["Simple as that.", "That matters.", "Think about it.", "Not a small thing."]
    beats_sw = ["Ni rahisi hivyo.", "Hilo ni muhimu.", "Fikiria hilo."]
    beats = beats_sw if lang == "sw" else beats_en
    out = []
    for i, s in enumerate(sentences):
        out.append(s)
        if i and i % 7 == 0 and rng.random() < 0.5:
            out.append(rng.choice(beats))
    return out


def humanize(text, level="medium", language="auto", seed=None):
    """Return (rewritten_text, list_of_changes)."""
    if level not in LEVELS:
        level = "medium"
    p = LEVELS[level]
    rng = random.Random(seed)

    if language == "auto":
        from detector import detect_language
        language = detect_language(text)

    paragraphs = re.split(r"\n{2,}", text.strip())
    result, changes = [], []
    for para in paragraphs:
        original = para
        para = _replace_phrases(para, rng, language, p)
        if language == "en":
            para = _contract(para, rng, p)
        sents = [s.strip() for s in SENT_SPLIT.split(para) if s.strip()]
        sents = _trim_transition_openers(sents, rng, p)
        sents = _split_long(sents, rng, p)
        sents = _merge_short(sents, rng, p * 0.6)
        if level == "strong":
            sents = _add_touches(sents, rng, language)
        para = " ".join(sents)
        para = re.sub(r"\s+([,.!?])", r"\1", para)
        para = re.sub(r"\s{2,}", " ", para)
        if para != original:
            changes.append(1)
        result.append(para)

    out = "\n\n".join(result)
    return out, {"level": level, "language": language,
                 "paragraphs_changed": len(changes)}


if __name__ == "__main__":
    import sys
    print(humanize(sys.stdin.read())[0])
