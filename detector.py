"""
detector.py - AI text detector (English + Kiswahili)

Combines stylometric signals with an optional language-model perplexity
signal (GPT-2). Output is a probability-like score 0-100, a label, the
individual signals and a per-sentence breakdown.

NOTE: Every AI detector is probabilistic. Treat the result as a hint,
not proof.
"""
import json
import math
import os
import re
import statistics
from collections import Counter

# ---------------------------------------------------------------- word lists
AI_PHRASES_EN = [
    "delve", "delves", "delving", "tapestry", "testament to", "in today's world",
    "in today's fast-paced", "in the realm of", "it is important to note",
    "it is worth noting", "it's worth noting", "plays a crucial role",
    "play a crucial role", "a myriad of", "multifaceted", "navigate the complexities",
    "in conclusion", "in summary", "to summarize", "moreover", "furthermore",
    "additionally", "ever-evolving", "rapidly evolving", "landscape of",
    "underscores", "underscore the", "pivotal", "seamless", "seamlessly",
    "leverage", "leveraging", "harness the power", "unlock the potential",
    "holistic", "robust", "foster", "fostering", "embark on", "journey",
    "cutting-edge", "game-changer", "shed light on", "a wide range of",
    "comprehensive", "ultimately", "overall,", "not only", "intricate",
    "paramount", "essential to", "crucial to", "vital role",
]

AI_PHRASES_SW = [
    "kwa kumalizia", "ni muhimu kutambua", "ni muhimu kuzingatia",
    "katika ulimwengu wa leo", "katika zama hizi", "kwa hakika",
    "isitoshe", "aidha", "zaidi ya hayo", "kwa ujumla", "hatimaye",
    "ina jukumu muhimu", "yana jukumu muhimu", "kuboresha ufanisi",
    "kwa kuzingatia hayo", "mchango mkubwa", "changamoto mbalimbali",
    "teknolojia ya kisasa", "maendeleo endelevu",
]

TRANSITIONS = [
    "moreover", "furthermore", "additionally", "however", "therefore", "thus",
    "consequently", "in addition", "as a result", "in conclusion", "overall",
    "nevertheless", "similarly", "likewise", "hence", "ultimately",
    "aidha", "hata hivyo", "kwa hiyo", "hivyo basi", "zaidi ya hayo",
    "kwa upande mwingine", "pia", "vilevile", "hatimaye",
]

CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n{2,}")
WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F']+")

SW_HINTS = {"na", "ya", "wa", "kwa", "ni", "katika", "za", "la", "hii", "huu",
            "kuwa", "cha", "vya", "pia", "sana", "hii", "yake", "wake", "lakini"}

# Kiswahili register markers (heuristic): formal/academic prose vs. everyday speech
SW_FORMAL = {"kwamba", "ambayo", "ambao", "ambaye", "ambapo", "ambazo", "ambalo",
             "kutokana", "kuzingatia", "mbalimbali", "husika", "ufanisi", "endelevu",
             "jamii", "mchango", "changamoto", "muhimu", "kuboresha", "wadau",
             "hivyo", "hususan", "kimsingi", "kikamilifu", "aidha", "isitoshe"}
SW_INFORMAL = {"yaani", "jamani", "poa", "basi", "bwana", "kabisa", "balaa", "daladala",
               "mambo", "eh", "ee", "aisee", "sijui", "tu", "hata", "ila", "bado",
               "kidogo", "leo", "jana", "kesho", "nimeona", "nikaona", "wacha"}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def split_sentences(text):
    parts = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s and s.strip()]
    return parts


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


SW_HINTS_EXTRA = {"kila", "kama", "mara", "yangu", "wote", "pale", "mbele", "kwenda",
                  "huwa", "hivyo", "alikuwa", "ilikuwa", "sijui", "kidogo", "lakini",
                  "ila", "basi", "yaani", "jamani", "kabisa", "mimi", "wewe", "sisi",
                  "wao", "mama", "baba", "rafiki", "kesho", "jana", "leo", "tu", "bado",
                  "kwamba", "ambayo", "wanafunzi", "walimu", "elimu", "hii", "hizi", "huu",
                  "mtu", "watu", "kazi", "nyumba", "shule", "sana", "zaidi", "kuwa"}
EN_HINTS = {"the", "and", "of", "to", "is", "in", "that", "it", "for", "with", "was",
            "are", "this", "on", "as", "be", "by", "or", "at", "from", "have", "not"}


def detect_language(text):
    ws = words(text)
    if not ws:
        return "en"
    sw = sum(1 for w in ws if w in SW_HINTS or w in SW_HINTS_EXTRA) / len(ws)
    en = sum(1 for w in ws if w in EN_HINTS) / len(ws)
    return "sw" if (sw > 0.10 and sw > en) else "en"


# --------------------------------------------------------- optional perplexity
_PPL = {"tried": False, "tok": None, "model": None}


def _load_lm():
    if _PPL["tried"]:
        return _PPL["model"] is not None
    _PPL["tried"] = True
    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _PPL["tok"] = GPT2TokenizerFast.from_pretrained("gpt2")
        _PPL["model"] = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        return True
    except Exception:
        return False


def perplexity(text):
    """GPT-2 perplexity, or None if torch/transformers are not installed."""
    if not _load_lm():
        return None
    import torch
    tok, model = _PPL["tok"], _PPL["model"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(math.exp(out.loss.item()))


# ------------------------------------------------------------------- signals
def _burstiness(sentences):
    lens = [len(words(s)) for s in sentences if words(s)]
    if len(lens) < 4:
        return None, None
    mean = statistics.mean(lens)
    cv = statistics.pstdev(lens) / mean if mean else 0
    # humans: cv about 0.55-0.9 ; AI: about 0.2-0.4
    return _clamp((0.55 - cv) / 0.3), cv


def _phrase_hits(text, lang):
    low = text.lower()
    phrases = AI_PHRASES_SW if lang == "sw" else AI_PHRASES_EN
    hits = [p for p in phrases if p in low]
    n_words = max(1, len(words(text)))
    per100 = len(hits) * 100 / n_words
    return _clamp(per100 / 2.5), hits


def _transition_density(sentences):
    if not sentences:
        return 0.0
    c = 0
    for s in sentences:
        head = s.lower().lstrip("\"'(")[:28]
        if any(head.startswith(t) for t in TRANSITIONS):
            c += 1
    rate = c / len(sentences)
    return _clamp((rate - 0.08) / 0.3), rate


def _contractions(text, lang):
    if lang != "en":
        return 0.5, None  # neutral
    n_words = len(words(text))
    if n_words < 120:
        return 0.5, None
    rate = len(CONTRACTION_RE.findall(text)) * 100 / n_words
    # no contractions across long text is a mild AI/formal sign
    return _clamp(1 - rate / 1.5), rate


def _opener_diversity(sentences):
    openers = [(words(s) or [""])[0] for s in sentences]
    if len(openers) < 5:
        return None
    top = Counter(openers).most_common(1)[0][1]
    uniq = len(set(openers)) / len(openers)
    return _clamp((0.8 - uniq) / 0.4 + (top / len(openers) - 0.2)), uniq


def _repetition(text):
    ws = words(text)
    if len(ws) < 60:
        return None
    tri = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    rep = 1 - len(set(tri)) / len(tri)
    return _clamp(rep / 0.08), rep


def _vocab_richness(text):
    ws = words(text)
    if len(ws) < 80:
        return None
    win = 50
    ttrs = [len(set(ws[i:i + win])) / win for i in range(0, len(ws) - win, 10)]
    m = statistics.mean(ttrs)
    # very uniform mid-high richness is typical of LLM prose; weak signal
    return _clamp(1 - abs(m - 0.72) / 0.2) * 0.6, m


def _informality(text):
    """Human text often has !, ?, ..., parentheses, dashes, lowercase 'i', typos."""
    n = max(1, len(split_sentences(text)))
    marks = len(re.findall(r"[!?]|\.\.\.|\(|\)|—|-{2}|;", text))
    return _clamp(1 - (marks / n) / 0.35), marks / n


def _sw_register(text):
    """Kiswahili: dense formal vocabulary and no colloquial markers -> more AI-like."""
    ws = words(text)
    n = max(1, len(ws))
    formal = sum(1 for w in ws if w in SW_FORMAL) * 100 / n
    informal = sum(1 for w in ws if w in SW_INFORMAL) * 100 / n
    score = _clamp(0.35 + formal / 12 - informal / 6)
    return score, formal, informal


def _sentence_score(s, lang):
    sc = 0.25
    low = s.lower()
    phrases = AI_PHRASES_SW if lang == "sw" else AI_PHRASES_EN
    if any(p in low for p in phrases):
        sc += 0.35
    if any(low.lstrip("\"'(").startswith(t) for t in TRANSITIONS):
        sc += 0.2
    n = len(words(s))
    if 18 <= n <= 30:
        sc += 0.1
    if n <= 5:
        sc -= 0.1
    if lang == "en" and CONTRACTION_RE.search(s):
        sc -= 0.1
    if re.search(r"[!?]", s):
        sc -= 0.1
    return _clamp(sc)


# -------------------------------------------------------------------- public
WEIGHTS = {
    "perplexity": 0.30,
    "burstiness": 0.18,
    "ai_phrases": 0.18,
    "transitions": 0.09,
    "contractions": 0.05,
    "opener_diversity": 0.07,
    "repetition": 0.05,
    "vocab_richness": 0.03,
    "informality": 0.05,
    "sw_register": 0.12,
}


FEATURE_KEYS = ["burstiness", "ai_phrases", "transitions", "contractions",
                "opener_diversity", "repetition", "vocab_richness", "informality",
                "sw_register"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
_MODEL = {"loaded": False, "data": None}


def feature_vector(signals):
    """Fixed-order vector for the trained classifier (missing signal -> 0.5)."""
    return [signals.get(k, 0.5) for k in FEATURE_KEYS]


def _model_probability(signals):
    """Probability (0-100) from model.json trained with train.py, else None."""
    if not _MODEL["loaded"]:
        _MODEL["loaded"] = True
        try:
            with open(MODEL_PATH, encoding="utf-8") as f:
                _MODEL["data"] = json.load(f)
        except (OSError, ValueError):
            _MODEL["data"] = None
    m = _MODEL["data"]
    if not m or m.get("keys") != FEATURE_KEYS:
        return None
    z = m["bias"] + sum(w * x for w, x in zip(m["weights"], feature_vector(signals)))
    z = max(-30.0, min(30.0, z))
    return 100 / (1 + math.exp(-z))


def analyze(text, use_perplexity=True):
    text = (text or "").strip()
    n_words = len(words(text))
    if n_words < 40:
        return {"error": "Weka angalau maneno 40 ili matokeo yawe na maana.",
                "word_count": n_words}

    lang = detect_language(text)
    sentences = split_sentences(text)
    signals, details = {}, {}

    s, cv = _burstiness(sentences)
    if s is not None:
        signals["burstiness"] = s
        details["burstiness"] = f"sentence-length variation = {cv:.2f}"

    s, hits = _phrase_hits(text, lang)
    signals["ai_phrases"] = s
    details["ai_phrases"] = hits

    s, rate = _transition_density(sentences)
    signals["transitions"] = s
    details["transitions"] = f"{rate * 100:.0f}% of sentences open with a transition"

    s, rate = _contractions(text, lang)
    signals["contractions"] = s

    r = _opener_diversity(sentences)
    if r:
        signals["opener_diversity"] = r[0]
        details["opener_diversity"] = f"{r[1] * 100:.0f}% unique sentence openers"

    r = _repetition(text)
    if r:
        signals["repetition"] = r[0]

    r = _vocab_richness(text)
    if r:
        signals["vocab_richness"] = r[0]

    s, per = _informality(text)
    signals["informality"] = s

    if lang == "sw":
        s, formal, informal = _sw_register(text)
        signals["sw_register"] = s
        details["sw_register"] = (f"formal words {formal:.1f}/100, "
                                  f"everyday/colloquial words {informal:.1f}/100")

    ppl = perplexity(text) if (use_perplexity and lang == "en") else None
    if ppl is not None:
        signals["perplexity"] = _clamp((70 - ppl) / 50)
        details["perplexity"] = f"GPT-2 perplexity = {ppl:.1f} (low = predictable = AI-like)"

    total_w = sum(WEIGHTS[k] for k in signals)
    score = sum(signals[k] * WEIGHTS[k] for k in signals) / total_w
    heuristic_pct = round(_clamp(score) * 100, 1)

    model_pct = _model_probability(signals)
    if model_pct is not None:
        score = 0.4 * score + 0.6 * (model_pct / 100)
        details["trained_model"] = f"trained classifier = {model_pct:.1f}% AI"
    score_pct = round(_clamp(score) * 100, 1)

    if score_pct >= 70:
        label = "Inawezekana sana imeandikwa na AI"
    elif score_pct >= 50:
        label = "Mchanganyiko / inashukiwa AI"
    elif score_pct >= 35:
        label = "Inaelekea binadamu, ina dalili chache za AI"
    else:
        label = "Inaelekea imeandikwa na binadamu"

    sent_out = [{"text": x, "ai_score": round(_sentence_score(x, lang) * 100)}
                for x in sentences]

    return {
        "ai_probability": score_pct,
        "label": label,
        "language": lang,
        "word_count": n_words,
        "sentence_count": len(sentences),
        "used_perplexity": ppl is not None,
        "signals": {k: round(v * 100) for k, v in signals.items()},
        "details": details,
        "sentences": sent_out,
        "disclaimer": "Detectors zote hukosea. Usitumie matokeo haya kama ushahidi pekee.",
    }


if __name__ == "__main__":
    import sys
    print(analyze(sys.stdin.read()))
