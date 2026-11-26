"""
train.py - train a small classifier on YOUR labelled data.

Folder layout:
    data/human/*.txt    texts written by people
    data/ai/*.txt       texts written by AI

Run:
    python train.py                 # writes model.json next to detector.py
    python train.py --data mydata --chunk 120

detector.py loads model.json automatically and blends it with the heuristics.
Pure Python (no numpy / sklearn needed).
"""
import argparse
import glob
import json
import math
import os
import random

from detector import FEATURE_KEYS, analyze, feature_vector, words


def chunks(text, size):
    ws = text.split()
    for i in range(0, len(ws), size):
        piece = ws[i:i + size]
        if len(piece) >= max(40, size // 2):
            yield " ".join(piece)


def load(folder, label, size):
    rows = []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for piece in chunks(f.read(), size):
                r = analyze(piece, use_perplexity=False)
                if "error" in r:
                    continue
                # analyze() returns percentages; convert back to 0-1
                sig = {k: v / 100 for k, v in r["signals"].items()}
                rows.append((feature_vector(sig), label, path))
    return rows


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1 / (1 + math.exp(-z))


def fit(train, epochs=400, lr=0.5, l2=0.01):
    n = len(train[0][0])
    w, b = [0.0] * n, 0.0
    for _ in range(epochs):
        gw, gb = [0.0] * n, 0.0
        for x, y, _p in train:
            err = sigmoid(b + sum(wi * xi for wi, xi in zip(w, x))) - y
            for j in range(n):
                gw[j] += err * x[j]
            gb += err
        m = len(train)
        w = [wi - lr * (g / m + l2 * wi) for wi, g in zip(w, gw)]
        b -= lr * gb / m
    return w, b


def accuracy(rows, w, b):
    if not rows:
        return float("nan")
    ok = sum((sigmoid(b + sum(wi * xi for wi, xi in zip(w, x))) >= 0.5) == bool(y)
             for x, y, _p in rows)
    return ok / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--chunk", type=int, default=120, help="words per training sample")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json"))
    a = ap.parse_args()

    rows = load(os.path.join(a.data, "human"), 0, a.chunk) + load(os.path.join(a.data, "ai"), 1, a.chunk)
    if len(rows) < 20:
        raise SystemExit(f"Only {len(rows)} samples found. Add more .txt files to {a.data}/human and {a.data}/ai.")

    # split by FILE so chunks of one document never land in both train and test
    files = sorted({r[2] for r in rows})
    random.Random(7).shuffle(files)
    test_files = set(files[:max(1, len(files) // 5)])
    train = [r for r in rows if r[2] not in test_files]
    test = [r for r in rows if r[2] in test_files]

    w, b = fit(train)
    print(f"samples: {len(rows)} (train {len(train)}, test {len(test)})")
    print(f"train accuracy: {accuracy(train, w, b):.2%}")
    print(f"test accuracy:  {accuracy(test, w, b):.2%}  <- trust this one")
    for k, wi in zip(FEATURE_KEYS, w):
        print(f"  {k:18s} {wi:+.2f}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"keys": FEATURE_KEYS, "weights": w, "bias": b}, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
