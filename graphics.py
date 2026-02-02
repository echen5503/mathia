from __future__ import annotations

import os
import math
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional: LLM
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_LLM = True
except Exception:
    HAS_LLM = False


# -----------------------------
# Output paths (ONLY what you need + loss demo)
# -----------------------------
BASE_DIR = "outputs"
WORDINFO_DIR = os.path.join(BASE_DIR, "word_info")
LOSS_DIR = os.path.join(BASE_DIR, "losses")
os.makedirs(WORDINFO_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)


# -----------------------------
# 1) Text preprocessing
# -----------------------------
def simple_word_tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    for p in [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "\"", "'"]:
        text = text.replace(p, f" {p} ")
    return [w for w in text.split(" ") if w != ""]


# -----------------------------
# 2) Filler method
# -----------------------------
FILLER_PROBS: Dict[str, float] = {
    "um": 0.95,
    "uh": 0.95,
    "like": 0.70,
    "so": 0.60,       # sentence start only
    "well": 0.60,
    "actually": 0.40,
    "basically": 0.40,
    "therefore": 0.05,
    "because": 0.05,
}
SENT_END = {".", "!", "?"}

def filler_I(words: List[str]) -> np.ndarray:
    I = np.zeros(len(words), dtype=float)
    i = 0
    sentence_start = True
    while i < len(words):
        w = words[i].lower()

        # "you know" bigram
        if i + 1 < len(words) and w == "you" and words[i + 1].lower() == "know":
            p = 0.80
            I[i] = 1.0 - p
            I[i + 1] = 1.0 - p
            i += 2
            sentence_start = False
            continue

        if sentence_start and w == "so":
            p = FILLER_PROBS.get("so", 0.0)
        else:
            p = FILLER_PROBS.get(w, 0.0)

        I[i] = 1.0 - p

        sentence_start = (words[i] in SENT_END)
        i += 1
    return I


# -----------------------------
# 3) Zip method
# -----------------------------
def compressed_len_bytes(s: str) -> int:
    return len(zlib.compress(s.encode("utf-8"), level=9))

def zip_I(words: List[str], k: float = 1.0, eps_i: float = 1e-9) -> np.ndarray:
    I = np.zeros(len(words), dtype=float)
    prev_c = 0
    for idx in range(len(words)):
        prefix = " ".join(words[: idx + 1])
        c = compressed_len_bytes(prefix)
        delta = c - prev_c
        correction = k * math.log(max(idx + 1, 2) + eps_i)
        I[idx] = delta * correction
        prev_c = c
    return I


# -----------------------------
# 4) LLM (GPT-2) surprisal (optional)
# -----------------------------
@dataclass
class GPT2Surprisal:
    tokenizer_name: str = "gpt2"
    model_name: str = "gpt2"
    device: str = "cuda" if HAS_LLM and torch.cuda.is_available() else "cpu"
    max_context: int = 512

    def __post_init__(self):
        if not HAS_LLM:
            raise RuntimeError("transformers/torch not installed.")
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def token_surprisals(self, text: str) -> Tuple[List[int], np.ndarray]:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)

        if input_ids.shape[1] > self.max_context:
            input_ids = input_ids[:, -self.max_context:]

        logits = self.model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        T = input_ids.shape[1]
        surps = torch.empty(T, device=self.device)
        surps[:] = float("nan")
        for t in range(1, T):
            tok_id = int(input_ids[0, t].item())
            lp = float(log_probs[0, t - 1, tok_id].item())
            surps[t] = -lp / math.log(2.0)  # bits

        return input_ids[0].tolist(), surps.detach().cpu().numpy()

    def word_surprisals(self, words: List[str]) -> np.ndarray:
        text = " ".join(words)
        token_ids, token_surps = self.token_surprisals(text)
        tokens = [self.tok.decode([tid]) for tid in token_ids]

        # word char spans
        spans = []
        pos = 0
        for w in words:
            if pos == 0:
                s, e = 0, len(w)
            else:
                s, e = pos + 1, pos + 1 + len(w)
            spans.append((s, e))
            pos = e

        Iw = np.zeros(len(words), dtype=float)
        char_offset = 0
        for t, tok_str in enumerate(tokens):
            ts, te = char_offset, char_offset + len(tok_str)
            if not np.isnan(token_surps[t]):
                for wi, (ws, we) in enumerate(spans):
                    if te > ws and ts < we:
                        Iw[wi] += float(token_surps[t])
            char_offset = te
        return Iw


# -----------------------------
# Normalization (same scale)
# -----------------------------
def minmax_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x)  # flat line if constant
    return (x - lo) / (hi - lo)


# -----------------------------
# Plot: per-word information (normalized)
# -----------------------------
def save_per_word_information(text: str, include_llm: bool = True) -> str:
    words = simple_word_tokenize(text)

    I_f = filler_I(words)
    I_z = zip_I(words, k=1.0)

    I_dict: Dict[str, np.ndarray] = {
        "Filler (norm)": minmax_01(I_f),
        "Zip (norm)": minmax_01(I_z),
    }

    if include_llm and HAS_LLM:
        try:
            llm = GPT2Surprisal()
            I_l = llm.word_surprisals(words)
            I_dict["LLM GPT-2 (norm)"] = minmax_01(I_l)
        except Exception as e:
            print("LLM disabled due to error:", e)

    plt.figure(figsize=(12, 4))
    x = np.arange(len(words))
    for name, I in I_dict.items():
        plt.plot(x, I, marker="o", linewidth=1.5, label=name)

    plt.xticks(x, words, rotation=60, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Information")
    plt.title("Per-word information contribution")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(WORDINFO_DIR, "per_word_information.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    return out_path


# -----------------------------
# Loss demo graphic (quick + visual)
# -----------------------------
def save_loss_demo_figure() -> str:
    # Pairwise comparison loss: 0-1 loss on a single pair as a function of margin m = Z(a)-Z(b)
    # If label y=1 (a should be more concise), then correct if m>0, incorrect if m<=0.
    m = np.linspace(-2, 2, 401)
    pair_loss_y1 = (m <= 0).astype(float)  # 1 if wrong, 0 if correct

    # Ordering loss: absolute displacement |pred_rank - true_rank| per item
    # Make a small illustrative example (n=8) with one big mistake
    true_rank = np.array([1,2,3,4,5,6,7,8])
    pred_rank = np.array([1,2,3,4,6,5,8,7])  # swaps + small errors
    disp = np.abs(pred_rank - true_rank)

    plt.figure(figsize=(11, 4))

    # Left panel: pairwise 0-1 loss vs margin
    plt.subplot(1, 2, 1)
    plt.plot(m, pair_loss_y1, linewidth=2)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Margin m = Z(a) - Z(b)")
    plt.ylabel("Pairwise loss (0 = correct, 1 = wrong)")
    plt.title("Pairwise comparison loss on one labeled pair\n(y=1: 'a more concise than b')")

    # Right panel: ordering loss as absolute displacement
    plt.subplot(1, 2, 2)
    x = np.arange(len(true_rank))
    plt.bar(x, disp)
    plt.xticks(x, [f"x{i+1}" for i in range(len(x))])
    plt.xlabel("Items")
    plt.ylabel("|pred_rank - true_rank|")
    plt.title("Ordering loss = sum of absolute rank errors\n(per-item displacement shown)")

    plt.tight_layout()

    out_path = os.path.join(LOSS_DIR, "loss_functions_demo.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    return out_path


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Use your sample text (same as before)
    text = "So, um, basically, the experiment shows a clear trend because the data is consistent."

    p1 = save_per_word_information(text, include_llm=True)
    p2 = save_loss_demo_figure()

    print("Saved:")
    print(" -", os.path.abspath(p1))
    print(" -", os.path.abspath(p2))
