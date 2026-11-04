"""Streamlit web interface for the Deepfake Detection System.

    streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import config

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.face_detector import FaceDetector, draw_faces
from src.forensics import error_level_analysis, frequency_spectrum

st.set_page_config(page_title="Deepfake Detection System", page_icon=":material/shield:",
                   layout="wide", initial_sidebar_state="expanded")

# Verdict colours (RGB for OpenCV drawing, hex for HTML/CSS).
REAL_RGB, FAKE_RGB = (22, 163, 74), (220, 38, 38)
REAL_HEX, FAKE_HEX, ACCENT = "#16A34A", "#DC2626", "#2563EB"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px;}

/* sidebar */
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E2E8F0;}
.side-brand {display: flex; align-items: center; gap: .7rem; margin-bottom: .4rem;}
.side-brand .logo {width: 40px; height: 40px; border-radius: 10px; display: grid; place-items: center;
    background: linear-gradient(135deg, #1D4ED8, #7C3AED); color: #fff; font-weight: 800;}
.side-brand .name {font-weight: 700; font-size: 1rem; line-height: 1.15; color: #0F172A;}
.side-brand .tag {font-size: .74rem; color: #64748B;}
.side-h {font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: #64748B; margin: 1.1rem 0 .3rem;}
.kv {width: 100%; font-size: .84rem; border-collapse: collapse; border: none !important;}
.kv tr, .kv td {border: none !important; background: transparent !important;}
.kv td {padding: .36rem 0 !important; border-bottom: 1px solid #F1F5F9 !important;}
.kv td:last-child {text-align: right; font-weight: 600; color: #0F172A;}
.kv td:first-child {color: #64748B;}

/* hero header */
.hero {background: linear-gradient(120deg, #0B1B3F 0%, #1E3A8A 55%, #4C1D95 100%);
    border-radius: 18px; padding: 1.7rem 2rem; color: #fff; margin-bottom: 1.2rem;
    display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; flex-wrap: wrap;
    box-shadow: 0 10px 30px -12px rgba(30, 58, 138, .45);}
.hero .eyebrow {font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; opacity: .75;}
.hero h1 {font-size: 1.85rem; font-weight: 800; margin: .25rem 0 .35rem; color: #fff; padding: 0;}
.hero p {margin: 0; opacity: .85; font-size: .98rem; max-width: 640px;}
.hero-stats {display: flex; gap: .7rem; flex-wrap: wrap;}
.hero-stat {background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
    border-radius: 12px; padding: .65rem 1rem; min-width: 112px; backdrop-filter: blur(4px);}
.hero-stat .v {font-size: 1.3rem; font-weight: 700;}
.hero-stat .l {font-size: .72rem; opacity: .8; text-transform: uppercase; letter-spacing: .06em;}

/* cards (st.container(key="card_*")) */
[class*="st-key-card"] {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
    padding: 1.2rem 1.3rem; box-shadow: 0 1px 3px rgba(15, 23, 42, .05);}
.card-title {font-size: 1rem; font-weight: 700; color: #0F172A; margin: 0 0 .15rem;}
.card-sub {font-size: .84rem; color: #64748B; margin: 0 0 .6rem;}

/* result */
.result {border-radius: 14px; padding: 1.2rem 1.3rem; border: 1px solid; margin-bottom: .9rem;}
.result.real {background: #F0FDF4; border-color: #BBF7D0;}
.result.fake {background: #FEF2F2; border-color: #FECACA;}
.badge {display: inline-flex; align-items: center; gap: .35rem; font-size: .72rem; font-weight: 700;
    letter-spacing: .07em; padding: .28rem .65rem; border-radius: 999px; text-transform: uppercase;}
.badge.real {background: #DCFCE7; color: #166534;}
.badge.fake {background: #FEE2E2; color: #991B1B;}
.result .label {font-size: 2rem; font-weight: 800; margin: .45rem 0 .1rem; line-height: 1.1;}
.result.real .label {color: #15803D;} .result.fake .label {color: #B91C1C;}
.result .desc {color: #334155; font-size: .93rem;}
.meter {position: relative; height: 10px; border-radius: 999px; background: #E2E8F0; margin: 1.1rem 0 .4rem;}
.meter .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}
.meter .thr {position: absolute; top: -5px; bottom: -5px; width: 2px; background: #0F172A; border-radius: 2px;}
.meter-l {display: flex; justify-content: space-between; font-size: .75rem; color: #64748B;}
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; margin-top: 1rem;}
.kpi {background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: .65rem .8rem;}
.kpi .v {font-size: 1.15rem; font-weight: 700; color: #0F172A;}
.kpi .l {font-size: .7rem; color: #64748B; text-transform: uppercase; letter-spacing: .04em;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.stat {margin-bottom: .8rem;} .stat .v {font-size: 1.5rem; font-weight: 700; color: #0F172A;}
.stat .l {font-size: .78rem; color: #64748B;}

/* face rows */
.face-h {display: flex; justify-content: space-between; align-items: center; margin-bottom: .4rem;}
.face-h .t {font-weight: 700; color: #0F172A;}
.pill {font-size: .75rem; font-weight: 600; padding: .2rem .6rem; border-radius: 999px;}
.pill.real {background: #DCFCE7; color: #166534;} .pill.fake {background: #FEE2E2; color: #991B1B;}

/* empty state / steps */
.steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem;}
.step {background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1rem;}
.step .n {width: 28px; height: 28px; border-radius: 8px; background: #EFF6FF; color: #1D4ED8;
    display: grid; place-items: center; font-weight: 700; font-size: .85rem; margin-bottom: .55rem;}
.step .t {font-weight: 700; color: #0F172A; font-size: .92rem;}
.step .d {color: #64748B; font-size: .82rem; margin-top: .2rem;}
.empty {text-align: center; padding: 2.2rem 1rem; color: #64748B;}
.empty .big {font-size: 1.05rem; font-weight: 600; color: #334155; margin-bottom: .25rem;}
.chips {display: flex; flex-wrap: wrap; gap: .45rem;}
.chip {background: #F1F5F9; border: 1px solid #E2E8F0; color: #334155; border-radius: 8px;
    padding: .3rem .65rem; font-size: .82rem; font-weight: 500;}
.thumb-cap {font-size: .76rem; text-align: center; margin-top: -.35rem; font-weight: 600;}
.footer {text-align: center; color: #94A3B8; font-size: .8rem; margin-top: 2rem;
    padding-top: 1rem; border-top: 1px solid #E2E8F0;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: .3rem; border-bottom: 1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"] {padding: .55rem 1rem; font-weight: 600;}
@media (max-width: 900px) {.kpis, .steps {grid-template-columns: repeat(2, 1fr);}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model and face detector ...")
def get_detector(model_path: str, backend: str):
    from src.predictor import DeepfakeDetector
    return DeepfakeDetector(model_path, backend)


def load_test_metrics() -> dict | None:
    path = config.REPORT_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def fmt_score(score: float | None) -> str:
    return "—" if score is None else f"{score:.1%}"


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def card_header(title: str, sub: str = ""):
    html(f'<div class="card-title">{title}</div>' + (f'<div class="card-sub">{sub}</div>' if sub else ""))


def result_card(res: dict, threshold: float, kind: str, kpis: list[tuple[str, str]]):
    cls = "fake" if res["is_fake"] else "real"
    badge = "Manipulation detected" if res["is_fake"] else "Verified authentic"
    desc = ("This content shows signs of being AI-generated or manipulated."
            if res["is_fake"] else "No signs of AI generation or manipulation were found.")
    p = res["fake_probability"]
    color = FAKE_HEX if res["is_fake"] else REAL_HEX
    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
                       for l, v in kpis)
    html(f"""
<div class="result {cls}">
  <span class="badge {cls}">{badge}</span>
  <div class="label">{res['label']} {kind}</div>
  <div class="desc">{desc}</div>
  <div class="meter">
    <div class="fill" style="width:{p * 100:.1f}%; background:{color};"></div>
    <div class="thr" style="left:calc({threshold * 100:.1f}% - 1px);" title="Decision threshold"></div>
  </div>
  <div class="meter-l"><span>Authentic</span>
    <span>Deepfake probability <b>{p:.1%}</b> · threshold {threshold:.0%}</span><span>Fake</span></div>
  <div class="kpis">{kpi_html}</div>
</div>""")


def empty_state(icon: str, title: str, text: str):
    html(f'<div class="empty"><div style="font-size:2.2rem">{icon}</div>'
         f'<div class="big">{title}</div><div>{text}</div></div>')


STEPS = [
    ("Face detection", "MTCNN locates every face and its landmarks."),
    ("Face extraction", "Faces are cropped with a margin and resized to 224 × 224."),
    ("CNN classification", "EfficientNet-B0 estimates the probability of manipulation."),
    ("Verdict & evidence", "Confidence score, Grad-CAM heatmap and forensic views."),
]


def steps_html() -> str:
    return '<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (t, d) in enumerate(STEPS, 1)) + "</div>"


def timeline_chart(df: pd.DataFrame, threshold: float) -> alt.Chart:
    df = df.assign(verdict=np.where(df["fake_probability"] >= threshold, "Deepfake", "Original"))
    base = alt.Chart(df).encode(
        x=alt.X("time:Q", title="Time (s)", axis=alt.Axis(grid=False, tickColor="#CBD5E1",
                                                          domainColor="#CBD5E1", tickCount=8)),
        y=alt.Y("fake_probability:Q", title="Deepfake probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", gridColor="#EEF2F7", domain=False, tickCount=5)),
    )
    area = base.mark_area(color=ACCENT, opacity=0.08)
    line = base.mark_line(color=ACCENT, strokeWidth=2)
    points = base.mark_circle(size=80, stroke="#FFFFFF", strokeWidth=2, opacity=1).encode(
        color=alt.Color("verdict:N", legend=None,
                        scale=alt.Scale(domain=["Original", "Deepfake"], range=[REAL_HEX, FAKE_HEX])),
        tooltip=[alt.Tooltip("time:Q", title="Time (s)", format=".2f"),
                 alt.Tooltip("frame:Q", title="Frame"),
                 alt.Tooltip("fake_probability:Q", title="Deepfake probability", format=".1%"),
                 alt.Tooltip("verdict:N", title="Frame verdict"),
                 alt.Tooltip("faces:Q", title="Faces")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        color="#475569", strokeDash=[5, 4], strokeWidth=1.5).encode(y="t:Q")
    label = alt.Chart(pd.DataFrame({"t": [threshold], "x": [df["time"].max()]})).mark_text(
        align="right", dy=-7, color="#475569", fontSize=11, fontWeight=600,
        text=f"Threshold {threshold:.0%}").encode(x="x:Q", y="t:Q")
    return (area + line + rule + label + points).properties(height=260, background="#FFFFFF").configure_view(
        strokeWidth=0).configure_axis(labelColor="#64748B", titleColor="#475569",
                                      labelFontSize=11, titleFontSize=12, titleFontWeight=600)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    html("""<div class="side-brand"><div class="logo">DF</div>
            <div><div class="name">Deepfake Detector</div>
            <div class="tag">Authenticity Verification</div></div></div>""")
    html('<div class="side-h">Detection</div>')
    backend = st.selectbox("Face detector", ["auto", *FaceDetector.available_backends()],
                           help="auto uses MTCNN, falling back to dlib if needed.")
    with st.expander("Advanced"):
        model_path = st.text_input("Model file", str(config.MODEL_PATH))

try:
    detector = get_detector(model_path, backend)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("**Getting started**\n\n"
            "1. Download a dataset: `python download_dataset.py`\n"
            "2. Train the model: `python train.py`\n3. Reload this page.")
    st.stop()

meta = detector.meta
test_metrics = load_test_metrics()

with st.sidebar:
    if backend != "auto" and detector.face_detector.backend != backend:
        st.warning(f"'{backend}' could not be loaded; using "
                   f"'{detector.face_detector.backend}' instead.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(detector.threshold), 0.01,
                          help="Media with a deepfake probability at or above this value is "
                               "classified as Deepfake. The default is calibrated on validation data.")
    n_frames = st.slider("Video frames to sample", 5, 60, config.FRAMES_PER_VIDEO, 5)
    html('<div class="side-h">Display</div>')
    show_cam = st.toggle("Grad-CAM heatmaps", True)
    show_forensics = st.toggle("Forensic analysis (ELA / FFT)", True)

    html('<div class="side-h">Model</div>')
    rows = [("Architecture", {"efficientnet": "EfficientNet-B0", "xception": "Xception",
                              "cnn": "Custom CNN"}.get(meta.get("arch"), meta.get("arch", "?"))),
            ("Input size", f"{meta['img_size']} × {meta['img_size']}"),
            ("Face detector", detector.face_detector.backend.upper()),
            ("Scene model", "Active" if detector.scene_model is not None else "Not trained")]
    if test_metrics:
        rows += [("Test accuracy", f"{test_metrics['accuracy']:.1%}"),
                 ("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")]
    html('<table class="kv">' + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
         + "</table>")

# ------------------------------------------------------------------- header
stats = []
if test_metrics:
    stats = [(f"{test_metrics['accuracy']:.1%}", "Test accuracy"),
             (f"{test_metrics['roc_auc']:.3f}", "ROC-AUC"),
             (f"{test_metrics['samples']:,}", "Test images")]
html(f"""
<div class="hero">
  <div>
    <div class="eyebrow">Deep Learning · Computer Vision</div>
    <h1>AI-Powered Deepfake Detection System</h1>
    <p>Verify the authenticity of images and videos. Upload media to find out whether it is
       original or AI-generated / manipulated, with a confidence score and visual evidence.</p>
  </div>
  <div class="hero-stats">{"".join(f'<div class="hero-stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in stats)}</div>
</div>""")

tab_img, tab_vid, tab_perf, tab_about = st.tabs([
    ":material/image: Image Analysis", ":material/movie: Video Analysis",
    ":material/monitoring: Model Performance", ":material/info: About"])

# -------------------------------------------------------------------- image
with tab_img:
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(key="card_img_upload"):
            card_header("Upload image", "JPG, PNG, BMP or WEBP · faces are detected automatically")
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                  label_visibility="collapsed")
            preview = st.empty()

    if not up:
        with right:
            with st.container(key="card_img_empty"):
                empty_state("🛡️", "No image analysed yet",
                            "Upload a photo on the left to start the authenticity check.")
        html('<div class="side-h" style="margin-top:1.4rem">How it works</div>' + steps_html())
    else:
        bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read this image. Please upload a valid image file.")
            st.stop()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with right, st.spinner("Analysing image ..."):
            res = detector.predict_image(rgb, threshold, explain=show_cam)

        n_faces = 0 if res["used_full_image"] else len(res["faces"])
        if res["used_full_image"]:
            preview.image(rgb, width="stretch")
        else:
            labels = [f"{f['label']} {f['confidence']:.0%}" for f in res["faces"]]
            colors = [FAKE_RGB if f["is_fake"] else REAL_RGB for f in res["faces"]]
            preview.image(draw_faces(rgb, res["faces"], labels, colors), width="stretch")

        with right:
            result_card(res, threshold, "Image", [
                ("Confidence", f"{res['confidence']:.1%}"),
                ("Face model", fmt_score(res["face_score"])),
                ("Scene model", fmt_score(res["scene_score"])),
                ("Faces", str(n_faces))])
            if res["used_full_image"] and res["scene_score"] is None:
                st.warning("No face was detected, so the whole image was analysed. The model is "
                           "trained on faces, so this result is less reliable.", icon=":material/warning:")
            elif res["used_full_image"]:
                st.info("No face was detected; the verdict comes from the whole-image (scene) model.",
                        icon=":material/info:")

            with st.container(key="card_faces"):
                card_header("Face-level analysis",
                            "Each detected face is classified independently. "
                            + ("Grad-CAM highlights the regions that influenced the decision."
                               if show_cam else ""))
                for i, f in enumerate(res["faces"]):
                    cls = "fake" if f["is_fake"] else "real"
                    html(f'<div class="face-h"><span class="t">Face {i + 1}</span>'
                         f'<span class="pill {cls}">{f["label"]} · {f["confidence"]:.0%}</span></div>')
                    cols = st.columns(3 if show_cam else 2, vertical_alignment="center")
                    cols[0].image(cv2.resize(f["crop"], (detector.img_size, detector.img_size)),
                                  caption="Face crop", width="stretch")
                    if show_cam:
                        cols[1].image(f["gradcam"], caption="Grad-CAM", width="stretch")
                    with cols[-1]:
                        c = FAKE_HEX if f["is_fake"] else REAL_HEX
                        html(f'<div class="stat"><div class="v" style="color:{c}">{f["fake_probability"]:.1%}</div>'
                             f'<div class="l">Deepfake probability</div></div>'
                             f'<div class="stat"><div class="v">{f["confidence"]:.1%}</div>'
                             f'<div class="l">Verdict confidence</div></div>')

        if show_forensics:
            with st.container(key="card_forensics"):
                card_header("Forensic analysis",
                            "Classical image-forensics views that support the model's verdict.")
                c1, c2 = st.columns(2, gap="large")
                c1.image(error_level_analysis(rgb), caption="Error Level Analysis (ELA)",
                         width="stretch")
                c1.caption("Regions that re-compress differently from the rest of the image "
                           "appear brighter and may indicate editing or synthesis.")
                c2.image(frequency_spectrum(res["faces"][0]["crop"]),
                         caption="Frequency spectrum of the face", width="stretch")
                c2.caption("Regular grid-like bright spots away from the centre are typical "
                           "up-sampling artifacts of GAN / diffusion generators.")

# -------------------------------------------------------------------- video
with tab_vid:
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(key="card_vid_upload"):
            card_header("Upload video", "MP4, AVI, MOV, MKV or WEBM · up to 500 MB")
            up_v = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                                    label_visibility="collapsed")
            if up_v:
                st.video(up_v.getvalue())

    if not up_v:
        with right:
            with st.container(key="card_vid_empty"):
                empty_state("🎬", "No video analysed yet",
                            f"Upload a video; {n_frames} evenly spaced frames will be analysed.")
    else:
        suffix = Path(up_v.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up_v.getvalue())
            video_path = tmp.name
        with right:
            bar = st.progress(0.0, text="Sampling frames and detecting faces ...")
            try:
                res = detector.predict_video(
                    video_path, n_frames, threshold,
                    progress_cb=lambda p: bar.progress(p, text=f"Analysing frames ... {p:.0%}"))
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            finally:
                Path(video_path).unlink(missing_ok=True)
            bar.empty()

            result_card(res, threshold, "Video", [
                ("Confidence", f"{res['confidence']:.1%}"),
                ("Fake frames", f"{res['fake_frame_ratio']:.0%}"),
                ("Face frames", f"{res['frames_with_faces']}/{res['frames_analyzed']}"),
                ("Jitter", f"{res['temporal_jitter']:.3f}")])
            v = res["video"]
            st.caption(f"Face model {fmt_score(res['face_score'])} · scene model "
                       f"{fmt_score(res['scene_score'])} (median over frames) · "
                       f"{v['width']}×{v['height']} · {v['fps']:.1f} fps · {v['duration']:.1f} s · "
                       "Temporal jitter = average change in deepfake probability between "
                       "consecutive frames; strong flicker is a common face-swap artifact.")
            if res["used_full_frames"]:
                st.warning("No faces were detected in the sampled frames; whole frames were analysed.",
                           icon=":material/warning:")

        with st.container(key="card_timeline"):
            card_header("Deepfake probability over time",
                        "Each point is one sampled frame (highest-scoring face). Hover for details.")
            df = pd.DataFrame(res["timeline"]).dropna(subset=["fake_probability"])
            if len(df):
                st.altair_chart(timeline_chart(df, threshold), width="stretch")

        with st.container(key="card_samples"):
            card_header("Most suspicious faces", "Sampled faces sorted by deepfake probability.")
            samples = sorted(res["samples"], key=lambda s: -s["fake_probability"])[:12]
            cols = st.columns(6)
            for i, s in enumerate(samples):
                with cols[i % 6]:
                    st.image(cv2.resize(s["crop"], (160, 160)), width="stretch")
                    c = FAKE_HEX if s["fake_probability"] >= threshold else REAL_HEX
                    html(f'<div class="thumb-cap" style="color:{c}">{s["fake_probability"]:.0%}'
                         f' <span style="color:#94A3B8;font-weight:500">· frame {s["frame"]}</span></div>')
            if show_cam and samples and not res["used_full_frames"]:
                st.divider()
                card_header("Grad-CAM of the most suspicious face")
                c1, c2, _ = st.columns([1, 1, 2])
                top = samples[0]["crop"]
                c1.image(cv2.resize(top, (detector.img_size, detector.img_size)), caption="Face",
                         width="stretch")
                c2.image(detector.explain(top), caption="Grad-CAM", width="stretch")

# -------------------------------------------------------------- performance
with tab_perf:
    if not test_metrics:
        st.info("Run `python evaluate.py` to generate test-set metrics and plots.")
    else:
        tiles = [("Accuracy", test_metrics["accuracy"]), ("Precision", test_metrics["precision"]),
                 ("Recall", test_metrics["recall"]), ("F1-score", test_metrics["f1_score"])]
        html('<div class="kpis" style="margin:0 0 1rem">' + "".join(
            f'<div class="kpi"><div class="v" style="font-size:1.6rem">{v:.1%}</div>'
            f'<div class="l">{l}</div></div>' for l, v in tiles) + "</div>")
        st.caption(f"Evaluated on {test_metrics['samples']:,} unseen {test_metrics['split']} images "
                   f"at threshold {test_metrics['threshold']:.2f} · ROC-AUC "
                   f"{test_metrics['roc_auc']:.4f}")
        c1, c2 = st.columns(2, gap="large")
        for col, key, title, file in [
            (c1, "card_cm", "Confusion matrix", "confusion_matrix.png"),
            (c2, "card_roc", "ROC curve", "roc_curve.png")]:
            path = config.REPORT_DIR / file
            if path.exists():
                with col, st.container(key=key):
                    card_header(title)
                    st.image(str(path), width="stretch")
        curves = config.REPORT_DIR / "training_curves.png"
        if curves.exists():
            with st.container(key="card_curves"):
                card_header("Training history", "Loss, accuracy and AUC on training and validation data.")
                st.image(str(curves), width="stretch")
        report = config.REPORT_DIR / "classification_report.txt"
        if report.exists():
            with st.expander("Classification report"):
                st.code(report.read_text(), language=None)

# -------------------------------------------------------------------- about
with tab_about:
    with st.container(key="card_about"):
        card_header("About the project")
        st.markdown(
            "Advances in generative AI make it possible to create highly realistic fake images "
            "and videos that are difficult for people to identify. This system automatically "
            "analyses uploaded media using **deep learning** and **computer vision**. It detects "
            "faces, examines them for visual inconsistencies and generation artifacts, and "
            "classifies the media as **Original** or **Deepfake** with a confidence score. "
            "It helps combat misinformation, identity fraud and the spread of fake digital content.")
        html('<div class="side-h">Detection pipeline</div>' + steps_html())
        html('<div class="side-h" style="margin-top:1.2rem">Technologies</div><div class="chips">'
             + "".join(f'<span class="chip">{t}</span>' for t in [
                 "Python", "TensorFlow / Keras", "OpenCV", "Streamlit", "CNN · EfficientNet-B0",
                 "MTCNN", "Dlib", "NumPy", "Pandas", "Scikit-learn", "Grad-CAM"]) + "</div>")
        html('<div class="side-h" style="margin-top:1.2rem">Limitations</div>')
        st.markdown(
            "- Accuracy depends on how similar the media is to the training data. New "
            "generation methods may evade detection.\n"
            "- Results are probabilistic and should support, not replace, human judgement.")

html('<div class="footer">AI-Powered Deepfake Detection System · Department of Information Technology, '
     'S R K R Engineering College (A), Bhimavaram · Project Work 2026–27</div>')
