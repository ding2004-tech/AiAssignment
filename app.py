import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

try:
    from deepface import DeepFace
    import deepface as _df_module
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,200,255,.08) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,.018) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,.018) 40px);
}

section[data-testid="stSidebar"] {
    background: #0c1220 !important;
    border-right: 1px solid rgba(0,200,255,.15) !important;
    padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] * { color: #c9d8ef !important; }

.sidebar-brand { display:flex; align-items:center; gap:.75rem; padding:0 1.25rem 1.8rem; border-bottom:1px solid rgba(0,200,255,.12); margin-bottom:1.4rem; }
.sidebar-brand-icon { width:40px; height:40px; background:linear-gradient(135deg,#00c8ff,#0070ff); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex-shrink:0; box-shadow:0 4px 16px rgba(0,200,255,.3); }
.sidebar-brand-text { font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:#fff !important; line-height:1.1; }
.sidebar-brand-sub { font-size:.68rem; color:#00c8ff !important; letter-spacing:.08em; text-transform:uppercase; }
.sidebar-section { font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:#4a6080 !important; padding:0 1.25rem .5rem; margin-top:.4rem; }

div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] > div { display:flex !important; flex-direction:column !important; gap:.3rem !important; padding:0 .75rem !important; }
div[data-testid="stRadio"] > div > label { display:flex !important; align-items:center !important; padding:.6rem 1rem !important; border-radius:10px !important; cursor:pointer !important; transition:all .2s ease !important; border:1px solid transparent !important; font-size:.88rem !important; font-weight:500 !important; color:#8aa0be !important; gap:.5rem !important; }
div[data-testid="stRadio"] > div > label > div:first-child { display:none !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) { background:rgba(0,200,255,.12) !important; border-color:rgba(0,200,255,.35) !important; color:#00c8ff !important; }
div[data-testid="stRadio"] > div > label:hover { background:rgba(0,200,255,.07) !important; color:#d0e8ff !important; border-color:rgba(0,200,255,.15) !important; }
div[data-testid="stRadio"] [data-baseweb="radio"] * { background:transparent !important; border-color:transparent !important; box-shadow:none !important; }
div[data-testid="stRadio"] input[type="radio"] { display:none !important; }

.stSelectbox > div > div, .stSlider > div { background:rgba(255,255,255,.04) !important; border-color:rgba(0,200,255,.2) !important; color:#c9d8ef !important; border-radius:10px !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#00c8ff !important; }
.stSlider [class*="StyledThumb"] { background:#00c8ff !important; }

.page-header { padding:2rem 0 1.5rem; }
.page-title { font-family:'Syne',sans-serif; font-weight:800; font-size:2rem; color:#fff; margin:0; letter-spacing:-.01em; }
.page-title span { color:#00c8ff; }
.page-subtitle { font-size:.88rem; color:#5a7090; margin-top:.35rem; font-weight:300; }

.stat-card { background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:1.1rem 1.3rem; position:relative; overflow:hidden; transition:border-color .2s; width:100%; }
.stat-card:hover { border-color:rgba(0,200,255,.3); }
.stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--accent,linear-gradient(90deg,#00c8ff,#0070ff)); }
.stat-num { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem; color:#fff; line-height:1; }
.stat-label { font-size:.75rem; color:#4a6080; text-transform:uppercase; letter-spacing:.1em; margin-top:.4rem; }
.stat-icon { position:absolute; top:1rem; right:1rem; font-size:1.4rem; opacity:.18; }

.upload-card { background:rgba(0,200,255,.04); border:1.5px dashed rgba(0,200,255,.25); border-radius:16px; padding:2rem; text-align:center; margin-bottom:1.2rem; }
.upload-card p { color:#5a7090; font-size:.85rem; margin:.5rem 0 0; }

.stButton > button { background:linear-gradient(135deg,#00c8ff 0%,#0070ff 100%) !important; color:#fff !important; border:none !important; border-radius:10px !important; padding:.65rem 1.8rem !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; font-size:.9rem !important; letter-spacing:.02em !important; cursor:pointer !important; transition:all .2s ease !important; box-shadow:0 4px 20px rgba(0,200,255,.25) !important; }
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 6px 28px rgba(0,200,255,.38) !important; }
.stButton > button:active { transform:translateY(0) !important; }

.result-panel { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:1.4rem; margin-top:1rem; }
.result-panel h4 { font-family:'DM Sans',sans-serif; font-weight:500; font-size:.82rem; color:#4a6080; margin:0 0 .9rem; letter-spacing:.04em; display:flex; align-items:center; gap:.4rem; }

.alert-danger { background:rgba(255,50,80,.1); border:1px solid rgba(255,50,80,.3); border-left:3px solid #ff3250; border-radius:10px; padding:.85rem 1.1rem; color:#ff8090; font-weight:500; font-size:.9rem; margin:.8rem 0; }
.alert-success { background:rgba(0,200,130,.1); border:1px solid rgba(0,200,130,.3); border-left:3px solid #00c882; border-radius:10px; padding:.85rem 1.1rem; color:#50e8a8; font-weight:500; font-size:.9rem; margin:.8rem 0; }
.alert-warning { background:rgba(255,180,0,.1); border:1px solid rgba(255,180,0,.3); border-left:3px solid #ffb400; border-radius:10px; padding:.85rem 1.1rem; color:#ffd060; font-weight:500; font-size:.9rem; margin:.8rem 0; }
.alert-info { background:rgba(0,200,255,.08); border:1px solid rgba(0,200,255,.25); border-left:3px solid #00c8ff; border-radius:10px; padding:.85rem 1.1rem; color:#80d8f0; font-weight:500; font-size:.9rem; margin:.8rem 0; }

.vio-card { background:rgba(255,50,80,.06); border:1px solid rgba(255,50,80,.18); border-radius:12px; padding:1rem 1.2rem; margin-bottom:.7rem; display:flex; align-items:center; gap:1rem; }
.vio-badge { background:rgba(255,50,80,.2); color:#ff6070; border-radius:8px; padding:.3rem .7rem; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; white-space:nowrap; }
.vio-name { font-weight:600; color:#d0e0f0; font-size:.9rem; }
.vio-time { font-size:.75rem; color:#4a6080; margin-top:.1rem; }

.stTextInput > div > div > input { background:rgba(255,255,255,.05) !important; border:1px solid rgba(255,255,255,.1) !important; border-radius:10px !important; color:#c9d8ef !important; font-family:'DM Sans',sans-serif !important; }
.stTextInput > div > div > input:focus { border-color:rgba(0,200,255,.5) !important; box-shadow:0 0 0 2px rgba(0,200,255,.12) !important; }

.model-badge { display:inline-flex; align-items:center; gap:.4rem; background:rgba(0,200,255,.1); border:1px solid rgba(0,200,255,.25); color:#00c8ff; font-size:.75rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; padding:.28rem .75rem; border-radius:20px; }
.model-badge-dot { width:6px; height:6px; border-radius:50%; background:#00c8ff; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.8);} }

.section-divider { border:none; border-top:1px solid rgba(255,255,255,.06); margin:1.5rem 0; }

.detection-pill { background:rgba(0,200,130,.12); border:1px solid rgba(0,200,130,.25); color:#40e090; font-size:.75rem; font-weight:700; padding:.2rem .6rem; border-radius:20px; }

.stImage img { border-radius:14px !important; border:1px solid rgba(255,255,255,.07) !important; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#080c14; }
::-webkit-scrollbar-thumb { background:#1a2a40; border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:#0070ff; }

.stFileUploader > div { background:rgba(255,255,255,.03) !important; border:1.5px dashed rgba(0,200,255,.22) !important; border-radius:14px !important; }
.stFileUploader label { color:#5a7090 !important; }
.stSlider label { color:#5a7090 !important; font-size:.78rem !important; }
.stSelectbox label { color:#5a7090 !important; font-size:.78rem !important; letter-spacing:.05em; }

/* Hide Streamlit metric default label styling */
[data-testid="stMetric"] { background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:1rem 1.3rem !important; }
[data-testid="stMetricValue"] { font-family:'Syne',sans-serif !important; font-weight:800 !important; font-size:1.9rem !important; color:#fff !important; }
[data-testid="stMetricLabel"] { font-size:.72rem !important; color:#4a6080 !important; text-transform:uppercase !important; letter-spacing:.1em !important; }

#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CLASS_NAMES = ["vape", "person", "smoke", "person2", "other"]

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "users" not in st.session_state:
    st.session_state["users"] = []
if "violations" not in st.session_state:
    st.session_state["violations"] = []

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M2 8C2 8 4 4 12 4C20 4 22 8 22 8" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
                <circle cx="12" cy="12" r="4" stroke="white" stroke-width="1.8"/>
                <circle cx="12" cy="12" r="1.5" fill="white"/>
                <path d="M2 16C2 16 4 20 12 20C20 20 22 16 22 16" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
        </div>
        <div>
            <div class="sidebar-brand-text">VisionGuard</div>
            <div class="sidebar-brand-sub">AI Detection System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    menu = st.radio(
        "Menu",
        ["🔍  Detection", "📊  Metrics", "🏢  Smart Surveillance", "👤  User Setup"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)

    model_option = st.selectbox("Active Model", ["YOLO", "RT-DETR", "Faster R-CNN"])
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.01)


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model(name):
    if name == "YOLO":
        return YOLO("FinalModel/best.pt")
    elif name == "RT-DETR":
        return YOLO("FinalModel/rtdetr.pt")
    else:
        num_classes = 5
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.2, 0.5, 1.0, 2.0, 5.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, trainable_backbone_layers=3)
        model.rpn.anchor_generator = rpn_anchor_generator
        model.rpn.head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load("FinalModel/smoke_model_v8.pth", map_location="cpu"))
        model.eval()
        return model

model = load_model(model_option)


# ─────────────────────────────────────────
#  UTIL FUNCTIONS
# ─────────────────────────────────────────
def convert_predictions(model_option, results):
    preds = []
    if model_option in ["YOLO", "RT-DETR"]:
        boxes  = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
    else:
        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
    for b, s, l in zip(boxes, scores, labels):
        preds.append({"box": b.tolist(), "score": float(s), "label": int(l)})
    return preds


def draw_boxes(image, preds):
    img = np.array(image).copy()
    color_map = {
        "vape":    (255, 80, 120),
        "person":  (0, 200, 255),
        "smoke":   (255, 160, 0),
        "person2": (100, 255, 180),
        "other":   (160, 130, 255),
    }
    for p in preds:
        if p["score"] > confidence:
            x1, y1, x2, y2 = map(int, p["box"])
            label = CLASS_NAMES[p["label"]]
            color = color_map.get(label, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label}  {p['score']:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(img, text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (20, 20, 30), 1)
    return img


def detect_violation(preds):
    has_person = any(CLASS_NAMES[p["label"]] in ("person", "person2") for p in preds if p["score"] > confidence)
    has_smoke  = any(CLASS_NAMES[p["label"]] in ("vape", "smoke")     for p in preds if p["score"] > confidence)
    return has_person and has_smoke


def crop_person(image, preds):
    img = np.array(image)
    best_area = 0
    best_crop = None
    for p in preds:
        if CLASS_NAMES[p["label"]] in ("person", "person2") and p["score"] > confidence:
            x1, y1, x2, y2 = map(int, p["box"])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_crop = img[y1:y2, x1:x2]
    return best_crop


def match_face(full_frame_array, preds):
    if not st.session_state["users"]:
        return {"name": "No Users", "status": "UNKNOWN", "distance": None, "confidence": None, "debug": "No users registered"}

    # ── Fallback: pixel MSE ──────────────────────────────────
    if not DEEPFACE_AVAILABLE:
        person_crop = crop_person(Image.fromarray(full_frame_array), preds)
        probe = person_crop if person_crop is not None else full_frame_array
        captured = cv2.resize(probe, (100, 100))
        best_diff = float("inf")
        best_name = "Unknown"
        for user in st.session_state["users"]:
            user_img = cv2.resize(np.array(user["image"]), (100, 100))
            diff = np.mean((captured.astype(float) - user_img.astype(float)) ** 2)
            if diff < best_diff:
                best_diff = diff
                best_name = user["name"]
        if best_diff < 2000:
            return {"name": best_name, "status": "MATCHED", "distance": round(best_diff, 2), "confidence": None, "debug": f"MSE={best_diff:.1f}"}
        return {"name": "Unknown", "status": "UNKNOWN", "distance": round(best_diff, 2), "confidence": None, "debug": f"MSE={best_diff:.1f} > 2000"}

    # ── FaceNet recognition ─────────────────────────────────────
    threshold = 0.60
    best_name     = "Unknown"
    best_distance = float("inf")
    debug_log     = []

    probes = []

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        p = f.name
        cv2.imwrite(p, cv2.cvtColor(full_frame_array, cv2.COLOR_RGB2BGR))
        probes.append(("full_frame", p))

    person_crop_arr = crop_person(Image.fromarray(full_frame_array), preds)
    if person_crop_arr is not None and person_crop_arr.size > 0:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            p = f.name
            cv2.imwrite(p, cv2.cvtColor(person_crop_arr, cv2.COLOR_RGB2BGR))
            probes.append(("person_crop", p))

    try:
        for user in st.session_state["users"]:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as gal_f:
                gal_path = gal_f.name
                user_bgr = cv2.cvtColor(np.array(user["image"]), cv2.COLOR_RGB2BGR)
                cv2.imwrite(gal_path, user_bgr)

            for probe_label, probe_path in probes:
                try:
                    emb1 = DeepFace.represent(
                        img_path          = probe_path,
                        model_name        = "Facenet",
                        detector_backend  = "opencv",
                        enforce_detection = False,
                        align             = True,
                    )
                    emb2 = DeepFace.represent(
                        img_path          = gal_path,
                        model_name        = "Facenet",
                        detector_backend  = "opencv",
                        enforce_detection = False,
                        align             = True,
                    )
                    vec1 = np.array(emb1[0]["embedding"])
                    vec2 = np.array(emb2[0]["embedding"])
                    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)
                    dist    = float(1.0 - cos_sim)
                    debug_log.append(f"{user['name']}@{probe_label}: {dist:.4f}")
                    if dist < best_distance:
                        best_distance = dist
                        best_name     = user["name"]
                except Exception as e:
                    debug_log.append(f"{user['name']}@{probe_label}: ERROR {str(e)[:80]}")

            os.unlink(gal_path)
    finally:
        for _, pp in probes:
            try:
                os.unlink(pp)
            except Exception:
                pass

    debug_str = " | ".join(debug_log)

    if best_distance <= threshold:
        confidence_pct = round((1 - best_distance / threshold) * 100, 1)
        return {
            "name":       best_name,
            "status":     "MATCHED",
            "distance":   round(best_distance, 4),
            "confidence": confidence_pct,
            "debug":      debug_str,
        }

    return {
        "name":       "Unknown",
        "status":     "UNKNOWN",
        "distance":   round(best_distance, 4) if best_distance != float("inf") else None,
        "confidence": None,
        "debug":      debug_str,
    }


def save_violation(name, image):
    st.session_state["violations"].append({
        "name":  name,
        "time":  datetime.now().strftime("%H:%M:%S"),
        "image": image,
    })


# ─────────────────────────────────────────
#  MODEL BADGE HELPER
# ─────────────────────────────────────────
MODEL_ICONS = {"YOLO": "⚡", "RT-DETR": "🔬", "Faster R-CNN": "🧠"}

def model_badge(name):
    icon = MODEL_ICONS.get(name, "🤖")
    return f'<span class="model-badge"><span class="model-badge-dot"></span>{icon} {name}</span>'


# ═══════════════════════════════════════════════════════════
#  🔍  DETECTION
# ═══════════════════════════════════════════════════════════
if menu == "🔍  Detection":
    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">Object <span>Detection</span></h1>
        <p class="page-subtitle">Upload an image to run inference with the active model</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(model_badge(model_option), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="result-panel"><h4>📁 Input Image</h4>', unsafe_allow_html=True)
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"],
                                label_visibility="collapsed")
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)
            w, h = image.size
            st.markdown(f"""
            <div style="display:flex;gap:.5rem;margin-top:.6rem;flex-wrap:wrap;">
                <span style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#5a7090;font-size:.72rem;padding:.2rem .6rem;border-radius:6px;">📐 {w} × {h}px</span>
                <span style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#5a7090;font-size:.72rem;padding:.2rem .6rem;border-radius:6px;">🖼 {file.type}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-card">
                <div style="font-size:2.5rem;opacity:.4;">🖼️</div>
                <p>Drag & drop or click to browse</p>
                <p style="font-size:.75rem;margin-top:.3rem;">JPG, PNG, JPEG supported</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="result-panel"><h4>🎯 Detection Result</h4>', unsafe_allow_html=True)
        if file:
            if st.button("▶  Run Detection", use_container_width=True):
                with st.spinner("Running inference…"):
                    if model_option in ["YOLO", "RT-DETR"]:
                        results = model(np.array(image))
                        preds = convert_predictions(model_option, results)
                    else:
                        img_tensor = T.ToTensor()(image)
                        with torch.no_grad():
                            prediction = model([img_tensor])[0]
                        preds = convert_predictions(model_option, prediction)

                result_img = draw_boxes(image, preds)
                st.image(result_img, use_container_width=True)

                filtered = [p for p in preds if p["score"] > confidence]
                st.markdown(f"""
                <div style="margin-top:.8rem;display:flex;gap:.5rem;flex-wrap:wrap;">
                    <span class="detection-pill">✅ {len(filtered)} detection{'s' if len(filtered)!=1 else ''}</span>
                </div>
                """, unsafe_allow_html=True)

                if filtered:
                    for p in filtered:
                        lbl   = CLASS_NAMES[p["label"]]
                        sc    = p["score"]
                        bar_w = int(sc * 100)
                        st.markdown(f"""
                        <div style="margin-top:.4rem;">
                            <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#8aa0be;">
                                <span>{lbl}</span><span>{sc:.0%}</span>
                            </div>
                            <div style="height:4px;background:rgba(255,255,255,.07);border-radius:2px;margin-top:.2rem;">
                                <div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#00c8ff,#0070ff);border-radius:2px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:200px;display:flex;align-items:center;justify-content:center;color:#2a3a50;font-size:.88rem;">
                Upload an image to see results
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  📊  METRICS
# ═══════════════════════════════════════════════════════════
elif menu == "📊  Metrics":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Model <span>Metrics</span></h1>
        <p class="page-subtitle">Evaluation results — Precision, Recall, F1-Score and Accuracy across all models</p>
    </div>
    """, unsafe_allow_html=True)

    METRICS_DATA = {
        "Model":     ["YOLO11x",  "RT-DETR", "Faster R-CNN + RPN"],
        "Accuracy":  [0.6871,     0.4671,    0.2367],
        "Precision": [0.7591,     0.5169,    0.4123],
        "Recall":    [0.8787,     0.8355,    0.5601],
        "F1-Score":  [0.8145,     0.6387,    0.4750],
    }
    MODELS   = METRICS_DATA["Model"]
    ICONS    = ["⚡", "🔬", "🧠"]
    COLORS   = ["#00c8ff", "#a78bfa", "#34d399"]
    BG       = "#080c14"
    PANEL_BG = "#0d1525"
    TEXT_DIM = "#4a6080"
    TEXT_MID = "#8aa0be"
    TEXT_LT  = "#c9d8ef"

    card_cols = st.columns(3, gap="medium")
    for i, (col, mdl, icon, color) in enumerate(zip(card_cols, MODELS, ICONS, COLORS)):
        acc  = METRICS_DATA["Accuracy"][i]
        prec = METRICS_DATA["Precision"][i]
        rec  = METRICS_DATA["Recall"][i]
        f1   = METRICS_DATA["F1-Score"][i]
        col.markdown(f"""
        <div style="background:{PANEL_BG};border:1px solid rgba(255,255,255,.07);border-top:2px solid {color};border-radius:14px;padding:1.3rem 1.4rem;">
            <div style="font-size:.72rem;color:{TEXT_DIM};letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem;">{icon} {mdl}</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:{color};line-height:1;">{acc*100:.2f}%</div>
            <div style="font-size:.72rem;color:{TEXT_DIM};margin-top:.25rem;">Accuracy</div>
            <div style="display:flex;gap:1rem;margin-top:.9rem;flex-wrap:wrap;">
                <div><div style="font-size:.8rem;font-weight:600;color:{TEXT_LT};">{prec*100:.2f}%</div><div style="font-size:.65rem;color:{TEXT_DIM};">Precision</div></div>
                <div><div style="font-size:.8rem;font-weight:600;color:{TEXT_LT};">{rec*100:.2f}%</div><div style="font-size:.65rem;color:{TEXT_DIM};">Recall</div></div>
                <div><div style="font-size:.8rem;font-weight:600;color:{TEXT_LT};">{f1*100:.2f}%</div><div style="font-size:.65rem;color:{TEXT_DIM};">F1-Score</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#c9d8ef;margin-bottom:.8rem;">📋 Comparison Table</div>
    """, unsafe_allow_html=True)

    METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-Score"]
    best_idx = {mk: METRICS_DATA[mk].index(max(METRICS_DATA[mk])) for mk in METRIC_KEYS}

    rows_html = ""
    for i, (mdl, icon, color) in enumerate(zip(MODELS, ICONS, COLORS)):
        cells = f"""<td style="padding:.75rem 1rem;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle;">
            <span style="display:inline-flex;align-items:center;gap:.4rem;">
                <span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0;"></span>
                <span style="font-weight:600;color:{TEXT_LT};">{icon} {mdl}</span>
            </span>
        </td>"""
        for mk in METRIC_KEYS:
            val     = METRICS_DATA[mk][i]
            is_best = (best_idx[mk] == i)
            badge   = (f"<span style='background:rgba(0,200,255,.12);color:#00c8ff;font-size:.62rem;font-weight:700;padding:.1rem .4rem;border-radius:4px;margin-left:.4rem;'>BEST</span>") if is_best else ""
            cells  += f"<td style='padding:.75rem 1rem;border-bottom:1px solid rgba(255,255,255,.04);font-weight:600;color:{color if is_best else TEXT_MID};'>{val*100:.2f}%{badge}</td>"
        rows_html += f"<tr style='transition:background .15s;' onmouseover=\"this.style.background='rgba(255,255,255,.025)'\" onmouseout=\"this.style.background=''\">{cells}</tr>"

    st.markdown(f"""
    <div style="background:{PANEL_BG};border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden;margin-bottom:1.8rem;">
        <table style="width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:.85rem;color:{TEXT_MID};">
            <thead>
                <tr style="background:{PANEL_BG};">
                    <th style="padding:.75rem 1rem;text-align:left;color:{TEXT_DIM};font-size:.7rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,.07);width:160px;">Model</th>
                    {''.join(f'<th style="padding:.75rem 1rem;text-align:left;color:{TEXT_DIM};font-size:.7rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,.07);">{mk}</th>' for mk in METRIC_KEYS)}
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#c9d8ef;margin-bottom:.8rem;">📈 Visual Comparison</div>
    """, unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2, gap="large")

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL_BG,
        "axes.edgecolor": "#1a2a40", "axes.labelcolor": TEXT_DIM,
        "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
        "text.color": TEXT_LT, "grid.color": "#1a2a40",
        "grid.linestyle": "--", "grid.linewidth": 0.6,
        "font.family": "sans-serif",
    })

    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x           = np.arange(len(MODELS))
        width       = 0.15
        bar_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        bar_colors  = ["#00c8ff", "#a78bfa", "#34d399", "#fb923c"]
        for j, (bm, bc) in enumerate(zip(bar_metrics, bar_colors)):
            vals = METRICS_DATA[bm]
            bars = ax.bar(x + j * width, vals, width, label=bm, color=bc, alpha=0.85, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, color=bc, fontweight="bold")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"{ic} {m}" for ic, m in zip(ICONS, MODELS)], fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.00])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title("All Metrics by Model", fontsize=10, fontweight="bold", color=TEXT_LT, pad=12)
        ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_MID, loc="lower right", ncol=2)
        ax.grid(axis="y", zorder=0)
        ax.spines[:].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_col2:
        categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
        N      = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
        ax.set_facecolor(PANEL_BG)
        fig.patch.set_facecolor(BG)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8, color=TEXT_MID)
        ax.set_ylim(0.0, 1.00)
        ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_yticklabels(["25%", "50%", "75%"], fontsize=6, color=TEXT_DIM)
        ax.grid(color="#1a2a40", linewidth=0.8)
        ax.spines["polar"].set_color("#1a2a40")
        for i, (mdl, color) in enumerate(zip(MODELS, COLORS)):
            vals = [METRICS_DATA[mk][i] for mk in categories]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=1.8, color=color, markersize=4, label=mdl)
            ax.fill(angles, vals, alpha=0.08, color=color)
        ax.set_title("Radar Comparison", fontsize=10, fontweight="bold", color=TEXT_LT, pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7.5, framealpha=0, labelcolor=TEXT_MID)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  🏢  SMART SURVEILLANCE
# ═══════════════════════════════════════════════════════════
elif menu == "🏢  Smart Surveillance":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Smart <span>Surveillance</span></h1>
        <p class="page-subtitle">Automated smoking / vaping detection with AI-powered identity recognition</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Live stats using st.metric (always re-renders correctly) ──
    total_v      = len(st.session_state["violations"])
    unique_names = len({v["name"] for v in st.session_state["violations"]})

    stat1, stat2, stat3 = st.columns(3, gap="medium")
    with stat1:
        st.markdown(f"""
        <div class="stat-card">
            <span class="stat-icon">🚨</span>
            <div class="stat-num">{total_v}</div>
            <div class="stat-label">Total Violations</div>
        </div>
        """, unsafe_allow_html=True)
    with stat2:
        st.markdown(f"""
        <div class="stat-card" style="--accent:linear-gradient(90deg,#ff6070,#ff3050);">
            <span class="stat-icon">💰</span>
            <div class="stat-num">RM{total_v * 300:,}</div>
            <div class="stat-label">Summons Issued</div>
        </div>
        """, unsafe_allow_html=True)
    with stat3:
        st.markdown(f"""
        <div class="stat-card" style="--accent:linear-gradient(90deg,#a070ff,#6030ff);">
            <span class="stat-icon">👤</span>
            <div class="stat-num">{unique_names}</div>
            <div class="stat-label">Unique Offenders</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Per-session duplicate guard ──────────────────────────────
    if "summoned_names" not in st.session_state:
        st.session_state["summoned_names"] = set()

    # ── Shared detection helper ───────────────────────────────────
    def run_detection_on_frame(frame_rgb, image_pil):
        if model_option in ["YOLO", "RT-DETR"]:
            results = model(frame_rgb)
            return convert_predictions(model_option, results)
        else:
            img_tensor = T.ToTensor()(image_pil)
            with torch.no_grad():
                prediction = model([img_tensor])[0]
            return convert_predictions(model_option, prediction)

    def show_identity_result(result, person_crop, frame_rgb, timestamp_str=""):
        """Render identity card. Skip summon if person already summoned this session."""
        if result["status"] == "MATCHED":
            name      = result["name"]
            conf_bar  = int(result["confidence"]) if result["confidence"] else 0
            dist_str  = f"{result['distance']}" if result["distance"] is not None else "—"
            label_extra = f" · {timestamp_str}" if timestamp_str else ""
            already_summoned = name in st.session_state["summoned_names"]

            st.markdown(f"""
            <div style="background:rgba(0,200,130,.08);border:1px solid rgba(0,200,130,.22);border-left:3px solid #00c882;border-radius:12px;padding:1rem 1.1rem;margin-top:.5rem;">
                <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;">
                    <span style="font-size:1.1rem;">✅</span>
                    <span style="font-weight:700;color:#40e090;font-size:.95rem;">Identity Matched{label_extra}</span>
                </div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#fff;margin-bottom:.5rem;">{name}</div>
                <div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:.6rem;">
                    <div style="background:rgba(0,200,130,.1);border-radius:8px;padding:.35rem .7rem;">
                        <div style="font-size:.65rem;color:#2a6048;text-transform:uppercase;letter-spacing:.07em;">Confidence</div>
                        <div style="font-weight:700;color:#40e090;font-size:.9rem;">{result['confidence']}%</div>
                    </div>
                    <div style="background:rgba(255,255,255,.05);border-radius:8px;padding:.35rem .7rem;">
                        <div style="font-size:.65rem;color:#4a6080;text-transform:uppercase;letter-spacing:.07em;">Distance</div>
                        <div style="font-weight:700;color:#8aa0be;font-size:.9rem;">{dist_str}</div>
                    </div>
                </div>
                <div style="height:4px;background:rgba(0,200,130,.15);border-radius:2px;">
                    <div style="width:{conf_bar}%;height:100%;background:linear-gradient(90deg,#00c882,#00e896);border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if already_summoned:
                st.markdown(f"""
                <div style="background:rgba(100,100,120,.08);border:1px solid rgba(100,100,120,.2);border-left:3px solid #556070;border-radius:10px;padding:.75rem 1rem;margin-top:.5rem;display:flex;align-items:center;gap:.6rem;">
                    <span style="font-size:1rem;">🔒</span>
                    <div style="font-size:.82rem;color:#6a8090;">
                        <strong style="color:#8aa0be;">{name}</strong> already summoned this session — no duplicate issued.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return False
            else:
                st.markdown(f"""
                <div style="background:rgba(255,180,0,.08);border:1px solid rgba(255,180,0,.22);border-left:3px solid #ffb400;border-radius:10px;padding:.85rem 1.1rem;margin-top:.5rem;display:flex;align-items:center;gap:.6rem;">
                    <span style="font-size:1.1rem;">💰</span>
                    <div>
                        <div style="font-weight:700;color:#ffd060;font-size:.9rem;">RM300 Summon Issued</div>
                        <div style="font-size:.75rem;color:#806020;margin-top:.1rem;">Issued to <strong style="color:#ffb400;">{name}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state["summoned_names"].add(name)
                save_violation(name, person_crop if person_crop is not None else frame_rgb)
                return True
        else:
            dist_str = f"{result['distance']}" if result["distance"] is not None else "inf"
            st.markdown(f"""
            <div style="background:rgba(255,180,0,.07);border:1px solid rgba(255,180,0,.2);border-left:3px solid #ffb400;border-radius:12px;padding:1rem 1.1rem;margin-top:.5rem;">
                <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem;">
                    <span style="font-size:1.1rem;">❓</span>
                    <span style="font-weight:700;color:#ffd060;font-size:.95rem;">Unknown Person</span>
                </div>
                <div style="font-size:.78rem;color:#806020;line-height:1.5;">
                    Face not found in the registered database.<br>
                    <span style="color:#4a6080;">Best distance: {dist_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return False

    # ── Single unified uploader (auto-detects image vs video) ───────
    col_feed, col_alert = st.columns([3, 2], gap="large")

    IMAGE_TYPES = ["jpg", "jpeg", "png"]
    VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]

    with col_feed:
        st.markdown('<div class="result-panel"><h4>📷 CCTV Feed</h4>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload image or video",
            type=IMAGE_TYPES + VIDEO_TYPES,
            label_visibility="collapsed",
        )

        # Detect file type
        is_video = False
        image_file = None
        video_file = None
        if uploaded:
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            is_video = ext in VIDEO_TYPES

        if uploaded and not is_video:
            image_file = uploaded
            image = Image.open(image_file).convert("RGB")
            frame = np.array(image)
            st.image(image, use_container_width=True)
            w, h = image.size
            st.markdown(f"""
            <div style="display:flex;gap:.5rem;margin-top:.6rem;flex-wrap:wrap;">
                <span style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#5a7090;font-size:.72rem;padding:.2rem .6rem;border-radius:6px;">🖼️ Image</span>
                <span style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#5a7090;font-size:.72rem;padding:.2rem .6rem;border-radius:6px;">📐 {w} × {h}px</span>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded and is_video:
            video_file = uploaded
            st.video(video_file)
            size_mb = video_file.size / (1024 * 1024)
            st.markdown(f"""
            <div style="display:flex;gap:.5rem;margin-top:.6rem;flex-wrap:wrap;">
                <span style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#5a7090;font-size:.72rem;padding:.2rem .6rem;border-radius:6px;">🎥 Video</span>
                <span style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#5a7090;font-size:.72rem;padding:.2rem .6rem;border-radius:6px;">📦 {size_mb:.1f} MB</span>
            </div>
            """, unsafe_allow_html=True)
            frame_interval    = st.slider("Analyse every N frames", 1, 60, 15,
                                          help="Lower = more thorough, slower")
            max_vio_per_video = st.slider("Max violations to log", 1, 20, 5)
        else:
            st.markdown("""
            <div class="upload-card">
                <div style="font-size:2.5rem;opacity:.3;">📷</div>
                <p>Upload an image or video — auto-detected</p>
                <p style="font-size:.75rem;">JPG · PNG · MP4 · AVI · MOV · MKV</p>
            </div>
            """, unsafe_allow_html=True)
            frame_interval    = 15
            max_vio_per_video = 5

        st.markdown("</div>", unsafe_allow_html=True)

    with col_alert:
        st.markdown('<div class="result-panel"><h4>⚠️ Alert System</h4>', unsafe_allow_html=True)

        # ── IMAGE ─────────────────────────────────────────────────────
        if image_file:
            if st.button("▶  Run Surveillance", use_container_width=True):
                with st.spinner("Analysing frame…"):
                    preds = run_detection_on_frame(frame, image)
                result_img = draw_boxes(image, preds)
                st.image(result_img, use_container_width=True)
                if detect_violation(preds):
                    st.markdown('<div class="alert-danger">🚨 &nbsp;Smoking / Vaping Detected!</div>', unsafe_allow_html=True)
                    person_crop = crop_person(image, preds)
                    if person_crop is not None:
                        st.markdown('<div style="font-size:.72rem;color:#4a6080;letter-spacing:.05em;margin:.6rem 0 .3rem;">CAPTURED PERSON</div>', unsafe_allow_html=True)
                        st.image(person_crop, use_container_width=True)
                    with st.spinner("🔍 Identifying person…"):
                        result = match_face(frame, preds)
                    if show_identity_result(result, person_crop if person_crop is not None else frame, frame):
                        st.rerun()
                else:
                    st.markdown('<div class="alert-info">✅ &nbsp;No violations detected in this frame.</div>', unsafe_allow_html=True)

        # ── VIDEO ─────────────────────────────────────────────────────
        elif video_file:
            if st.button("▶  Analyse Video", use_container_width=True):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name

                cap          = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps          = cap.get(cv2.CAP_PROP_FPS) or 25

                progress_bar  = st.progress(0, text="Starting analysis…")
                frame_display = st.empty()
                status_box    = st.empty()

                frame_idx   = 0
                vio_count   = 0
                needs_rerun = False

                while cap.isOpened():
                    ret, bgr = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    progress = min(frame_idx / max(total_frames, 1), 1.0)
                    time_sec = frame_idx / fps
                    progress_bar.progress(progress,
                        text=f"Frame {frame_idx}/{total_frames} — {time_sec:.1f}s | Violations: {vio_count}")

                    if frame_idx % frame_interval != 0:
                        continue

                    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pil   = Image.fromarray(rgb)
                    preds = run_detection_on_frame(rgb, pil)

                    annotated = draw_boxes(pil, preds)
                    frame_display.image(annotated, use_container_width=True,
                                        caption=f"⏱ {time_sec:.1f}s — Frame {frame_idx}")

                    if detect_violation(preds) and vio_count < max_vio_per_video:
                        vio_count += 1
                        status_box.markdown(
                            f'<div class="alert-danger">🚨 Violation at {time_sec:.1f}s (frame {frame_idx})</div>',
                            unsafe_allow_html=True)
                        person_crop = crop_person(pil, preds)
                        with st.spinner("🔍 Identifying person…"):
                            result = match_face(rgb, preds)
                        if show_identity_result(result, person_crop, rgb, timestamp_str=f"{time_sec:.1f}s"):
                            needs_rerun = True

                cap.release()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                progress_bar.progress(1.0,
                    text=f"✅ Done — {vio_count} violation(s) across {frame_idx} frames")
                if vio_count == 0:
                    status_box.markdown(
                        '<div class="alert-info">✅ No violations detected in this video.</div>',
                        unsafe_allow_html=True)
                if needs_rerun:
                    st.rerun()

        # ── EMPTY STATE ───────────────────────────────────────────────
        else:
            st.markdown("""
            <div style="height:200px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.6rem;color:#2a3a50;">
                <div style="font-size:2rem;opacity:.25;">📡</div>
                <div style="font-size:.85rem;">Upload an image or video to begin</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Violation log
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:#c9d8ef;margin-bottom:.8rem;">🗂️ Violation Log</div>
    """, unsafe_allow_html=True)

    if not st.session_state["violations"]:
        st.markdown("""
        <div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.5rem;text-align:center;color:#2a3a50;font-size:.85rem;">
            No violations recorded yet
        </div>
        """, unsafe_allow_html=True)
    else:
        for v in reversed(st.session_state["violations"]):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"""
                <div class="vio-card">
                    <span style="font-size:1.5rem;">🚨</span>
                    <div>
                        <div class="vio-name">{v['name']}</div>
                        <div class="vio-time">🕐 {v['time']}</div>
                    </div>
                    <span class="vio-badge">RM300 Summon</span>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.image(v["image"], width=120)


# ═══════════════════════════════════════════════════════════
#  👤  USER SETUP
# ═══════════════════════════════════════════════════════════
elif menu == "👤  User Setup":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">User <span>Setup</span></h1>
        <p class="page-subtitle">Register individuals for automated identity matching during surveillance</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_list = st.columns([5, 6], gap="large")

    with col_form:
        st.markdown("""
        <div style="background:#0d1525;border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:1.6rem 1.8rem;">
            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,.06);">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,#00c8ff,#0070ff);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.95rem;flex-shrink:0;">👤</div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;color:#c9d8ef;">Register New User</div>
                    <div style="font-size:.72rem;color:#4a6080;margin-top:.05rem;">Add a person to the recognition database</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        name       = st.text_input("Full Name", placeholder="e.g. Ahmad Faiz bin Abdullah")
        image_file = st.file_uploader("Face Photo", type=["jpg", "png", "jpeg"])

        if image_file:
            prev_img = Image.open(image_file).convert("RGB")
            prev_col, _ = st.columns([1, 2])
            with prev_col:
                st.markdown("""
                <div style="margin:.6rem 0 .4rem;font-size:.72rem;color:#4a6080;letter-spacing:.04em;">PREVIEW</div>
                """, unsafe_allow_html=True)
                st.image(prev_img, use_container_width=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        if st.button("＋  Register User", use_container_width=True):
            if name and image_file:
                image = Image.open(image_file).convert("RGB")
                st.session_state["users"].append({"name": name, "image": image})
                st.markdown(f"""
                <div style="background:rgba(0,200,130,.08);border:1px solid rgba(0,200,130,.25);border-left:3px solid #00c882;border-radius:10px;padding:.9rem 1.1rem;margin-top:.6rem;display:flex;align-items:center;gap:.7rem;">
                    <span style="font-size:1.2rem;">✅</span>
                    <div>
                        <div style="color:#40e090;font-weight:600;font-size:.9rem;">Successfully Registered</div>
                        <div style="color:#2a6048;font-size:.78rem;margin-top:.1rem;"><strong style="color:#50c890;">{name}</strong> has been added to the database.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(255,180,0,.08);border:1px solid rgba(255,180,0,.25);border-left:3px solid #ffb400;border-radius:10px;padding:.9rem 1.1rem;margin-top:.6rem;display:flex;align-items:center;gap:.7rem;">
                    <span style="font-size:1.2rem;">⚠️</span>
                    <div style="color:#ffd060;font-size:.88rem;font-weight:500;">Please provide both a full name and a face photo.</div>
                </div>
                """, unsafe_allow_html=True)

    with col_list:
        user_count = len(st.session_state["users"])
        st.markdown(f"""
        <div style="background:#0d1525;border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:1.6rem 1.8rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,.06);">
                <div style="display:flex;align-items:center;gap:.6rem;">
                    <div style="width:32px;height:32px;background:rgba(0,200,255,.1);border:1px solid rgba(0,200,255,.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.95rem;">👥</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;color:#c9d8ef;">Registered Users</div>
                        <div style="font-size:.72rem;color:#4a6080;margin-top:.05rem;">Identity recognition database</div>
                    </div>
                </div>
                <div style="background:rgba(0,200,255,.1);border:1px solid rgba(0,200,255,.2);color:#00c8ff;font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;">{user_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if user_count == 0:
            st.markdown("""
            <div style="background:#0d1525;border:1px solid rgba(255,255,255,.05);border-radius:14px;padding:3rem 1rem;text-align:center;margin-top:.5rem;">
                <div style="font-size:2.5rem;opacity:.2;margin-bottom:.8rem;">👤</div>
                <div style="color:#2a3a50;font-size:.85rem;line-height:1.6;">No users registered yet.<br><span style="color:#1e3050;">Add your first person using the form.</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for idx, u in enumerate(st.session_state["users"]):
                img_col, info_col = st.columns([1, 4])
                with img_col:
                    st.image(u["image"], use_container_width=True)
                with info_col:
                    st.markdown(f"""
                    <div style="padding:.4rem 0;">
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;color:#c9d8ef;margin-bottom:.3rem;">{u['name']}</div>
                        <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;">
                            <span style="background:rgba(0,200,130,.1);border:1px solid rgba(0,200,130,.2);color:#40e090;font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:.2rem .55rem;border-radius:20px;">● Registered</span>
                            <span style="background:rgba(0,200,255,.07);border:1px solid rgba(0,200,255,.15);color:#4a8090;font-size:.65rem;padding:.2rem .55rem;border-radius:20px;">ID #{idx+1:03d}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('<div style="height:1px;background:rgba(255,255,255,.04);margin:.4rem 0 .7rem;"></div>', unsafe_allow_html=True)
