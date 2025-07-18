#!/usr/bin/env python
"""
game_titlewordcloud_blue2green.py

Generate a YouTube‑style (tall rounded rectangle, triangle blank) word‑cloud
from /mnt/data/game_2024-2025.csv.

Colour gradient: BLUE (top) ➜ GREEN (bottom)
Cleaning: emoji stripped, non‑ASCII stripped, tokens <=2 chars removed
Stopwords: platform + gaming clutter

Output: youtube_game_wordcloud_blue2green.png
"""

# -------------------------------------------------------- imports
import os, sys, re, numpy as np, pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# -------------------------------------------------------- dataset path
CSV_PATH = "game_2024-2025.csv"   # <<< change if needed

# Optional explicit title column name; set to string to override auto‑detect
TITLE_COL_OVERRIDE = None

OUT_PNG = "youtube_game_wordcloud_blue2green.png"

# -------------------------------------------------------- geometry (taller rectangle)
W, H        = 1600, 900
MARGIN_X    = int(W * 0.10)   # 10% side padding
MARGIN_Y    = int(H * 0.12)   # 12% top/bottom padding -> taller rectangle
RECT_RADIUS = int(H * 0.20)   # rounded corners

TRI_W_FACTOR = 0.28           # triangle width % of rect width
TRI_H_FACTOR = 0.40           # triangle height % of rect height

# -------------------------------------------------------- filtering
MIN_WORD_LEN = 3

# Gaming / platform noise stopwords (extend as you see patterns)
CUSTOM_STOPWORDS = {
    # platform-y
    "official", "video", "full", "hd", "live", "new", "watch",
    # gaming clutter
    "gameplay", "game", "games", "gaming", "playthrough", "walkthrough",
    "dlc", "beta", "alpha", "update", "patch",
    "trailer", "trailers", "launch", "release",
    "highlight", "highlights", "clip", "clips",
    "match", "matches", "ranked", "rank",
    "episode", "ep", "pt", "part",
    "short", "shorts", "#shorts",
    # years
    "2024", "2025",
}
STOPWORDS_ALL = STOPWORDS | CUSTOM_STOPWORDS

# -------------------------------------------------------- regexes
BASE_RE   = re.compile(r"https?://\S+|www\.\S+|<[^>]+>|[@#]\w+")
EMOJI_RE  = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF" "\U000024C2-\U0001F251" "]+",
    flags=re.UNICODE
)
NON_WORD_RE = re.compile(r"[^\w\s]")

# -------------------------------------------------------- column detection
def detect_title_col(df: pd.DataFrame) -> str:
    """
    Try override; else common gaming title labels; else col containing 'title'.
    """
    if TITLE_COL_OVERRIDE:
        if TITLE_COL_OVERRIDE in df.columns:
            return TITLE_COL_OVERRIDE
        raise ValueError(
            f"TITLE_COL_OVERRIDE '{TITLE_COL_OVERRIDE}' not in columns: {list(df.columns)}"
        )

    candidates = ["title", "game_title", "video_title", "name", "game", "clip_title"]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col

    hits = [c for c in df.columns if "title" in c.lower()]
    if hits:
        return hits[0]

    raise ValueError(
        "Could not detect a game title column automatically. "
        f"Available columns: {list(df.columns)}. "
        "Set TITLE_COL_OVERRIDE at top of script."
    )

# -------------------------------------------------------- cleaning
def clean_and_filter(text: str) -> str:
    """Clean raw title -> filtered token string (len>=MIN_WORD_LEN, ascii, no stopwords)."""
    if not isinstance(text, str):
        return ""
    t = BASE_RE.sub(" ", text)
    t = EMOJI_RE.sub(" ", t)
    t = NON_WORD_RE.sub(" ", t)
    tokens = [
        tok.lower()
        for tok in re.split(r"\s+", t)
        if len(tok) >= MIN_WORD_LEN
    ]
    tokens = [
        tok.encode("ascii", "ignore").decode("ascii", "ignore")
        for tok in tokens
        if tok not in STOPWORDS_ALL
    ]
    return " ".join(tokens)

def build_corpus(series: pd.Series) -> str:
    return " ".join(series.dropna().map(clean_and_filter).tolist())

# -------------------------------------------------------- mask
def create_mask() -> np.ndarray:
    """
    mask == 0 -> draw zone (rounded rectangle)
    mask == 255 -> block (triangle + outside)
    """
    mask_img = Image.new("L", (W, H), 255)
    draw     = ImageDraw.Draw(mask_img)

    left, right = MARGIN_X, W - MARGIN_X
    top,  bot   = MARGIN_Y, H - MARGIN_Y
    draw.rounded_rectangle([(left, top), (right, bot)],
                           radius=RECT_RADIUS, fill=0)

    tri_w = int((right - left) * TRI_W_FACTOR)
    tri_h = int((bot   - top)  * TRI_H_FACTOR)
    cx, cy = W // 2, H // 2
    triangle = [
        (cx - tri_w // 2, cy - tri_h // 2),
        (cx - tri_w // 2, cy + tri_h // 2),
        (cx + tri_w // 2, cy),
    ]
    draw.polygon(triangle, fill=255)
    return np.array(mask_img)

# -------------------------------------------------------- BLUE -> GREEN gradient
# Using slightly softened blue & green to avoid eye‑searing pure primaries.
BLUE_RGB  = (  0,  64, 255)   # deepish blue
GREEN_RGB = (  0, 200,   0)   # strong green, not neon

def blue_to_green(word, font_size, position, orientation, random_state=None, **kw):
    """
    Interpolate RGB from BLUE_RGB (top) to GREEN_RGB (bottom) by vertical position.
    """
    _, y = position
    rect_height = H - 2 * MARGIN_Y
    ratio = (y - MARGIN_Y) / rect_height
    ratio = max(0.0, min(1.0, ratio))  # clamp

    r = int(BLUE_RGB[0]  + (GREEN_RGB[0]  - BLUE_RGB[0])  * ratio)
    g = int(BLUE_RGB[1]  + (GREEN_RGB[1]  - BLUE_RGB[1])  * ratio)
    b = int(BLUE_RGB[2]  + (GREEN_RGB[2]  - BLUE_RGB[2])  * ratio)
    return (r, g, b)

# -------------------------------------------------------- main
def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    title_col = detect_title_col(df)
    print(f"Using title column: {title_col}")

    corpus = build_corpus(df[title_col])
    if not corpus.strip():
        sys.exit("No usable words after cleaning/filtering; check title column and filters.")

    mask = create_mask()

    wc = WordCloud(
        width=W,
        height=H,
        background_color="white",
        mask=mask,
        stopwords=STOPWORDS_ALL,
        collocations=False,
        color_func=blue_to_green,
        prefer_horizontal=0.9,
    ).generate(corpus)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    wc.to_file(OUT_PNG)
    print(f"Saved game word‑cloud → {OUT_PNG}")

# -------------------------------------------------------- entry‑point
if __name__ == "__main__":
    main()
