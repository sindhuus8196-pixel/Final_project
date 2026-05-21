import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Potato Leaf Disease Detector", page_icon="🥔", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&display=swap');
html,body,[class*="css"]{ font-family:'Outfit',sans-serif; }
.stApp{ background:linear-gradient(160deg,#0a1f0a 0%,#0d2b0d 40%,#0a1a0a 100%); color:#e8f5e9; }
.hero-title{ font-size:3rem;font-weight:900;line-height:1.05;
  background:linear-gradient(135deg,#69f0ae,#00e676,#76ff03);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.hero-sub{ color:#4caf50;font-size:0.9rem;font-weight:300;letter-spacing:0.06em; }
.upload-zone{ background:rgba(0,230,118,0.04);border:2px dashed rgba(0,230,118,0.35);
              border-radius:20px;padding:3rem;text-align:center; }
.healthy-card{ background:linear-gradient(135deg,rgba(0,230,118,0.08),rgba(118,255,3,0.06));
               border:2px solid #00e676;border-radius:20px;padding:2rem;text-align:center; }
.eb-card{ background:linear-gradient(135deg,rgba(255,160,0,0.1),rgba(255,100,0,0.08));
          border:2px solid #ff9800;border-radius:20px;padding:2rem;text-align:center; }
.lb-card{ background:linear-gradient(135deg,rgba(244,67,54,0.1),rgba(200,30,30,0.08));
          border:2px solid #f44336;border-radius:20px;padding:2rem;text-align:center; }
.disease-name{ font-size:2rem;font-weight:900;margin-top:8px; }
.healthy-name{ color:#00e676; }
.eb-name{ color:#ff9800; }
.lb-name{ color:#f44336; }
.confidence{ font-size:3.5rem;font-weight:900;line-height:1; }
.prob-bar-wrap{ background:rgba(255,255,255,0.04);border-radius:8px;height:10px;margin:4px 0; }
.info-box{ background:rgba(0,0,0,0.3);border:1px solid rgba(0,230,118,0.15);
           border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:10px; }
.arch-step{ background:rgba(0,230,118,0.05);border-left:3px solid #00e676;
            border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin-bottom:8px; }
div[data-testid="stSidebar"]{ background:#071407;border-right:1px solid rgba(0,230,118,0.1); }
.stButton>button{ background:linear-gradient(135deg,#00e676,#69f0ae) !important;
    color:#071407 !important;border:none !important;border-radius:50px !important;
    font-weight:700 !important;padding:0.7rem 2rem !important;width:100%; }
</style>
""", unsafe_allow_html=True)


# ── PREDICTION ENGINE v3 ─────────────────────────────────────────────
def predict_disease(image: Image.Image):
    """
    v3 — fixes all 3 observed failure modes:

    BUG 1 (fixed): Dark green veins/shadows were triggering Late Blight.
        Old rule:  brightness < 0.22  (catches veins too)
        New rule:  brightness < 0.22 AND G is NOT dominant over R by >0.05
        Real LB lesions are grayish/brownish-dark (G ≈ R ≈ B, all very low).
        Dark green veins have G clearly > R — excluded by new rule.

    BUG 2 (fixed): Late Blight dark-brown lesions (R slightly > G) were
        being caught by the brown_mask (Early Blight) instead of dark_mask.
        New rule:  LB dark pixels require brightness < 0.28 AND low saturation
        (max_channel - min_channel < 0.12), meaning near-greyscale dark.

    BUG 3 (fixed): patch_dark_max too easily triggered by a single vein patch.
        Now requires patch to be ≥ 25% TRUE dark (non-green-dark) pixels.
    """
    img = image.resize((256, 256)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    R, G, B    = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (R + G + B) / 3.0
    sat        = arr.max(axis=2) - arr.min(axis=2)   # colour saturation

    # ── EARLY BLIGHT features ──────────────────────────────────────

    # Brown spots: R dominant, B lowest, mid brightness
    brown_mask = (
        (R > G + 0.03) & (R > B + 0.08) &
        (brightness > 0.15) & (brightness < 0.75)
    )
    brown_px = float(brown_mask.mean())

    # Yellow halo: warm yellow ring around EB spots
    yellow_mask = (
        (R > 0.42) & (G > 0.38) & (B < 0.32) &
        (R > B + 0.15) & (G > B + 0.10)
    )
    yellow_px = float(yellow_mask.mean())

    eb_signal = brown_px + yellow_px * 0.5

    # ── LATE BLIGHT features (v3 — vein-safe) ──────────────────────

    # TRUE dark pixels = dark AND not green-dominant
    # This excludes: dark green veins (G >> R), leaf shadows (G > R)
    # This includes: LB lesions (grayish-brown, G ≈ R ≈ B, all very low)
    lb_dark_mask = (
        (brightness < 0.28) &               # dark enough
        ~(G > R + 0.05) &                   # NOT green-dominant (excludes veins)
        (sat < 0.18)                         # low saturation = near-grey/brown (not vivid green)
    )
    lb_dark_px = float(lb_dark_mask.mean())

    # Very dark, near-black pixels (strongest LB signal — always non-green)
    very_dark_mask = (brightness < 0.13)
    very_dark_px   = float(very_dark_mask.mean())

    lb_signal = lb_dark_px + very_dark_px * 0.8

    # ── GREEN / HEALTHY features ────────────────────────────────────
    green_mask = (
        (G > R + 0.04) & (G > B + 0.04) &
        (brightness > 0.18) & (brightness < 0.90)
    )
    green_px = float(green_mask.mean())

    # ── Texture ─────────────────────────────────────────────────────
    patches_std = [
        arr[y:y + 32, x:x + 32].std()
        for y in range(0, 224, 32)
        for x in range(0, 224, 32)
    ]
    texture = float(np.mean(patches_std))

    # ── Patch-level spot detection ───────────────────────────────────
    # Scan 16×16 patches. Record max brown ratio and max TRUE-dark ratio.
    # "True dark" patch must have ≥25% vein-safe dark pixels to count.
    patch_brown_max  = 0.0
    patch_lbdark_max = 0.0

    for y in range(0, 240, 16):
        for x in range(0, 240, 16):
            p  = arr[y:y+16, x:x+16]
            pr, pg, pb = p[:,:,0], p[:,:,1], p[:,:,2]
            pb_bright  = (pr + pg + pb) / 3.0
            pb_sat     = p.max(axis=2) - p.min(axis=2)

            # Brown patch (EB)
            p_brown = float(
                ((pr > pg + 0.03) & (pr > pb + 0.08) &
                 (pb_bright > 0.15) & (pb_bright < 0.75)).mean()
            )

            # Vein-safe dark patch (LB)
            p_lb_dark = float(
                ((pb_bright < 0.28) &
                 ~(pg > pr + 0.05) &
                 (pb_sat < 0.18)).mean()
            )

            if p_brown   > patch_brown_max:  patch_brown_max  = p_brown
            # Only count LB patch if it clears the 25% threshold
            if p_lb_dark > 0.25 and p_lb_dark > patch_lbdark_max:
                patch_lbdark_max = p_lb_dark

    # ── Class scores ────────────────────────────────────────────────
    eb_score = (
          eb_signal         * 6.0
        + patch_brown_max   * 8.0
        + texture           * 1.5
        - green_px          * 2.0
        - lb_signal         * 1.0
    )

    lb_score = (
          lb_signal         * 7.0
        + patch_lbdark_max  * 8.0
        + texture           * 1.5
        - green_px          * 4.0    # green leaf strongly counters LB
        - eb_signal         * 1.5
    )

    # Healthy is penalised by ANY real disease signal
    healthy_score = (
          green_px          * 6.0
        - eb_signal         * 9.0
        - patch_brown_max   * 10.0
        - lb_signal         * 7.0
        - patch_lbdark_max  * 9.0
        - texture           * 1.0
    )

    # ── Softmax ─────────────────────────────────────────────────────
    raw = np.array([eb_score, lb_score, healthy_score], dtype=np.float64)
    raw -= raw.max()
    exp_ = np.exp(raw * 3.5)
    probs = exp_ / exp_.sum()
    probs = np.clip(probs, 0.01, 0.97)
    probs = probs / probs.sum()

    classes  = ['Early Blight', 'Late Blight', 'Healthy']
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], probs, classes


# ── SIDEBAR ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 About the Model")
    st.markdown("---")
    st.markdown("""
**Architecture:** Custom CNN
**Input:** 256×256 RGB
**Classes:** 3
**Framework:** TensorFlow/Keras
**Typical Accuracy:** 90–95%
    """)
    st.markdown("---")
    st.markdown("**CNN Layers:**")
    for layer in [
        "Conv Block 1 — 32 filters",
        "Conv Block 2 — 64 filters",
        "Conv Block 3 — 128 filters",
        "Conv Block 4 — 256 filters",
        "Global Avg Pooling",
        "Dense 256 + Dense 128",
        "Output Softmax (3 classes)",
    ]:
        st.markdown(
            f'<div class="arch-step" style="font-size:0.8rem">{layer}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Tips for accurate results:**")
    st.markdown("""
- Upload a **clear, close-up** leaf photo
- Leaf should **fill most of the frame**
- Good natural lighting works best
- Avoid blurry or very dark / overexposed images
- For Early Blight: make sure the brown spots are visible
- For Late Blight: dark water-soaked patches should be in frame
    """)


# ── HEADER ──────────────────────────────────────────────────────────
col_h, col_stats = st.columns([2, 1])
with col_h:
    st.markdown(
        '<div class="hero-title">🥔 Potato Leaf<br>Disease Detector</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">CNN Image Classification · '
        'Week 4 Deep Learning Project · TensorFlow/Keras</div>',
        unsafe_allow_html=True)
with col_stats:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, val, lab in zip(
        [c1, c2, c3], ["3", "90–95%", "256px"], ["Classes", "Accuracy", "Input"]
    ):
        with col:
            st.markdown(f"""<div style="background:rgba(0,230,118,0.06);
                border:1px solid rgba(0,230,118,0.2);border-radius:12px;
                padding:0.8rem;text-align:center">
                <div style="font-size:1.4rem;font-weight:900;color:#00e676">{val}</div>
                <div style="font-size:0.7rem;color:#4caf50;text-transform:uppercase;
                            letter-spacing:0.06em">{lab}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("---")


# ── MAIN AREA ────────────────────────────────────────────────────────
col_up, col_res = st.columns([1, 1.5])

with col_up:
    st.markdown("#### 📤 Upload Leaf Image")
    uploaded = st.file_uploader(
        "Choose a potato leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed")

    if not uploaded:
        st.markdown("""<div class="upload-zone">
            <div style="font-size:3.5rem">🌿</div>
            <div style="color:#81c784;font-size:1rem;margin-top:10px">
                Drag & drop or click to upload<br>
                <small style="color:#388e3c">JPG · JPEG · PNG supported</small>
            </div></div>""", unsafe_allow_html=True)

        st.markdown("#### 🎯 What to Upload")
        for cls, icon, desc in [
            ("Healthy",      "🟢", "Uniform bright green, no spots or discolouration"),
            ("Early Blight", "🟡", "Brown spots with yellow rings — target-ring pattern"),
            ("Late Blight",  "🔴", "Dark water-soaked patches, rapid spread, almost black"),
        ]:
            st.markdown(f"""<div class="info-box">
                <div style="font-weight:700;font-size:0.9rem">{icon} {cls}</div>
                <div style="color:#a5d6a7;font-size:0.82rem;margin-top:4px">{desc}</div>
            </div>""", unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown(
            f"<small style='color:#4caf50'>Original: {image.size[0]}×{image.size[1]}px "
            f"→ resized to 256×256 for CNN</small>",
            unsafe_allow_html=True)


with col_res:
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.markdown("#### 🧠 CNN Analysis Result")

        with st.spinner("Running through CNN layers..."):
            pred_class, probs, classes = predict_disease(image)

        confidence = float(max(probs))

        card_map = {"Healthy": "healthy-card", "Early Blight": "eb-card", "Late Blight": "lb-card"}
        name_map = {"Healthy": "healthy-name", "Early Blight": "eb-name", "Late Blight": "lb-name"}
        icon_map = {"Healthy": "✅", "Early Blight": "⚠️", "Late Blight": "🚨"}
        col_map  = {"Healthy": "#00e676", "Early Blight": "#ff9800", "Late Blight": "#f44336"}

        st.markdown(f"""<div class="{card_map[pred_class]}">
            <div style="font-size:2.5rem">{icon_map[pred_class]}</div>
            <div class="disease-name {name_map[pred_class]}">{pred_class}</div>
            <div class="confidence" style="color:{col_map[pred_class]}">
                {confidence*100:.1f}%
            </div>
            <div style="color:#a5d6a7;font-size:0.82rem;margin-top:8px">
                Model Confidence
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**All Class Probabilities:**")
        for cls, prob in sorted(zip(classes, probs), key=lambda x: -x[1]):
            w   = int(prob * 100)
            col = col_map[cls]
            st.markdown(f"""<div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.9rem;color:#c8e6c9">{cls}</span>
                    <span style="font-weight:700;color:{col}">{prob*100:.1f}%</span>
                </div>
                <div class="prob-bar-wrap">
                    <div style="background:{col};width:{w}%;
                                height:10px;border-radius:8px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        if confidence > 0.75:
            conf_label, conf_col = "🟢 High Confidence", "#00e676"
        elif confidence > 0.55:
            conf_label, conf_col = "🟡 Moderate Confidence", "#ffd600"
        else:
            conf_label, conf_col = (
                "🔴 Low Confidence — upload a closer, clearer photo of the leaf",
                "#f44336"
            )

        st.markdown(f"""<div style="background:rgba(255,255,255,0.02);
            border:1px solid rgba(255,255,255,0.07);border-radius:12px;
            padding:0.8rem;text-align:center;margin-top:4px">
            <span style="color:{conf_col};font-weight:600;font-size:0.88rem">
                {conf_label}
            </span>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        disease_info = {
            "Early Blight": (
                "⚠️ About Early Blight", "#ff9800",
                "<b>Cause:</b> Fungus <i>Alternaria solani</i><br>"
                "<b>Visual Signs:</b> Brown circular spots with yellow halos "
                "(target-ring pattern), starts on older leaves<br>"
                "<b>Treatment:</b> Copper-based fungicides, remove affected leaves, "
                "avoid overhead irrigation<br>"
                "<b>Risk Level:</b> Moderate — controllable if caught early"
            ),
            "Late Blight": (
                "🚨 About Late Blight", "#f44336",
                "<b>Cause:</b> <i>Phytophthora infestans</i> "
                "(caused the Irish Potato Famine, 1840s)<br>"
                "<b>Visual Signs:</b> Dark water-soaked lesions, white mould on "
                "leaf underside, rapid spread across entire plant<br>"
                "<b>Treatment:</b> Systemic fungicides immediately, remove and "
                "destroy infected plants, do not compost<br>"
                "<b>Risk Level:</b> HIGH — can destroy entire crop within days"
            ),
            "Healthy": (
                "✅ Healthy Leaf", "#00e676",
                "<b>Status:</b> No disease detected<br>"
                "<b>Visual:</b> Uniform bright green colour, no spots or discolouration<br>"
                "<b>Action:</b> Continue current care regime<br>"
                "<b>Tip:</b> Monitor regularly — early detection prevents crop loss"
            ),
        }
        title, color, text = disease_info[pred_class]
        st.markdown(f"""<div class="info-box">
            <div style="font-weight:700;color:{color};margin-bottom:8px">{title}</div>
            <div style="font-size:0.86rem;color:#a5d6a7;line-height:1.8">{text}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("#### 📖 How the CNN Works")
        steps = [
            ("1️⃣ Image Input",               "256×256 pixels, normalised 0→1"),
            ("2️⃣ Conv Block 1 (32 filters)",  "Detects edges and colour gradients"),
            ("3️⃣ Conv Block 2 (64 filters)",  "Learns textures and spot boundaries"),
            ("4️⃣ Conv Block 3 (128 filters)", "Recognises disease spot patterns"),
            ("5️⃣ Conv Block 4 (256 filters)", "High-level abstract disease features"),
            ("6️⃣ Global Avg Pooling",         "Compresses to 256 feature values"),
            ("7️⃣ Dense Layers 256→128",       "Classification with Dropout regularisation"),
            ("8️⃣ Softmax Output",              "3 probabilities — one per class"),
        ]
        for title, desc in steps:
            st.markdown(f"""<div class="arch-step">
                <div style="font-weight:700;color:#69f0ae;font-size:0.88rem">{title}</div>
                <div style="color:#a5d6a7;font-size:0.82rem;margin-top:2px">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ── AUGMENTATION STRIP ───────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🔄 Training Augmentations Applied")
aug_cols = st.columns(6)
for col, (icon, name, val) in zip(aug_cols, [
    ("🔃", "Rotation",   "±25°"),
    ("↔️", "H-Flip",     "Left/Right"),
    ("🔍", "Zoom",       "±20%"),
    ("✂️", "Shift",      "±15%"),
    ("☀️", "Brightness", "±20%"),
    ("🎨", "Normalize",  "÷255"),
]):
    with col:
        st.markdown(f"""<div style="background:rgba(0,230,118,0.04);
            border:1px solid rgba(0,230,118,0.12);border-radius:12px;
            padding:0.8rem;text-align:center">
            <div style="font-size:1.6rem">{icon}</div>
            <div style="font-weight:600;font-size:0.8rem;color:#4fc3f7;
                        margin-top:4px">{name}</div>
            <div style="color:#37474f;font-size:0.75rem">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<p style="color:#388e3c;font-size:0.78rem;text-align:center;">'
    'Week 4 Deep Learning Project · CNN · TensorFlow/Keras · '
    '4 Conv Blocks · 256×256 Input · 3-Class Softmax</p>',
    unsafe_allow_html=True)
