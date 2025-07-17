#!/usr/bin/env python
"""
YouTube‑style word‑cloud (v8, taller rectangle):
– White background
– Words in a red → dark‑yellow gradient
– Words fill a *taller* rounded rectangle (triangle blank)
– Emoji / non‑ASCII stripped
– Drops tokens with length ≤ 2
Output → youtube_logo_wordcloud_rect_red2darkyellow.png
"""

# -------------------------------------------------------- imports
import os, sys, re, numpy as np, pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# -------------------------------------------------------- paths / columns
CSV_PATH  = "sports_2024-2025_with_popularity.csv"   # ← change if needed
TITLE_COL = None                                               # set if auto‑detect fails

OUT_PNG   = "youtube_logo_wordcloud_rect_red2darkyellow.png"

# -------------------------------------------------------- geometry  (⬇️ vertical margin ↓)
W, H        = 1600, 900
MARGIN_X    = int(W * 0.10)   # 10 % left‑right padding  (unchanged)
MARGIN_Y    = int(H * 0.12)   # **12 % top‑bottom padding** (was 20 %)
RECT_RADIUS = int(H * 0.20)   # corner radius unchanged

TRI_W_FACTOR = 0.28           # triangle width  % of rect width
TRI_H_FACTOR = 0.40           # triangle height % of rect height

# -------------------------------------------------------- filtering
MIN_WORD_LEN = 3
CUSTOM_STOPWORDS = {
    "short", "shorts", "#shorts", "official", "video", "full", "hd",
    "live", "new", "watch", "episode", "ep", "pt", "part",
    "highlight", "highlights", "match", "matches",
    "reaction", "reactions", "review", "reviews", "trailer", "trailers",
    "game", "games", "2024", "2025"
}
STOPWORDS_ALL = STOPWORDS | CUSTOM_STOPWORDS

# -------------------------------------------------------- helpers
def detect_title_col(df, preferred):
    if preferred and preferred in df.columns:
        return preferred
    hits = [c for c in df.columns if "title" in c.lower()]
    if not hits:
        raise ValueError("No column with 'title' found.  Set TITLE_COL.")
    return hits[0]

BASE_RE   = re.compile(r"https?://\S+|www\.\S+|<[^>]+>|[@#]\w+")
EMOJI_RE  = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
NON_WORD_RE = re.compile(r"[^\w\s]")

def clean_and_filter(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = BASE_RE.sub(" ", text)
    t = EMOJI_RE.sub(" ", t)
    t = NON_WORD_RE.sub(" ", t)
    tokens = [tok.lower()
              for tok in re.split(r"\s+", t)
              if len(tok) >= MIN_WORD_LEN]
    tokens = [tok.encode("ascii", "ignore").decode("ascii", "ignore")
              for tok in tokens if tok not in STOPWORDS_ALL]
    return " ".join(tokens)

def create_mask() -> np.ndarray:
    mask_img = Image.new("L", (W, H), 255)
    draw     = ImageDraw.Draw(mask_img)

    left, right = MARGIN_X, W - MARGIN_X
    top,  bot   = MARGIN_Y, H - MARGIN_Y
    draw.rounded_rectangle([(left, top), (right, bot)],
                           radius=RECT_RADIUS, fill=0)

    tri_w = int((right - left) * TRI_W_FACTOR)
    tri_h = int((bot   - top)  * TRI_H_FACTOR)
    cx, cy = W // 2, H // 2
    triangle = [(cx - tri_w // 2, cy - tri_h // 2),
                (cx - tri_w // 2, cy + tri_h // 2),
                (cx + tri_w // 2, cy)]
    draw.polygon(triangle, fill=255)
    return np.array(mask_img)

def red_to_darkyellow(word, font_size, position, orientation, random_state=None, **kw):
    _, y = position
    rect_height = H - 2 * MARGIN_Y
    ratio = (y - MARGIN_Y) / rect_height
    ratio = max(0.0, min(1.0, ratio))
    r = 255
    g = int(180 * ratio)     # 0 → 180
    b = 0
    return (r, g, b)

# -------------------------------------------------------- main
def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    title_col = detect_title_col(df, TITLE_COL)
    print(f"Using column: {title_col}")

    corpus = " ".join(df[title_col].dropna().map(clean_and_filter).tolist())
    mask   = create_mask()

    wc = WordCloud(
        width=W,
        height=H,
        background_color="white",
        mask=mask,
        stopwords=STOPWORDS_ALL,
        collocations=False,
        color_func=red_to_darkyellow,
        prefer_horizontal=0.9,
    ).generate(corpus)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    wc.to_file(OUT_PNG)
    print(f"Saved to {OUT_PNG}")

if __name__ == "__main__":
    main()
