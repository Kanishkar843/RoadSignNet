"""Run with: streamlit run app.py"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from roadsignnet.artifacts import load_bundle
from roadsignnet.predict import predict_image

st.set_page_config(page_title="RoadSignNet · Kanishkar B", page_icon="🚦", layout="wide")
st.markdown("""<style>
.stApp {background: #0c1423; color: #e7edf8;}
[data-testid="stSidebar"] {background: #121f32;}
.block-container {max-width: 1100px; padding-top: 3rem;}
.eyebrow {color: #52dbb0; letter-spacing: .15em; font-size: .8rem; font-weight: 700;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">DEEP LEARNING & COMPUTER VISION</div>', unsafe_allow_html=True)
st.title("RoadSignNet")
st.write("Explore what a convolutional neural network sees in a traffic sign.")
st.caption("Kanishkar B · Traffic Sign Classification Using Deep Learning and Computer Vision")

with st.sidebar:
    st.header("Prediction settings")
    threshold = st.slider("Low-confidence threshold", 0.0, 1.0, 0.7, 0.05)
    st.caption("This marks low scores for review. A high score does not establish correctness.")
    st.divider()
    st.markdown("**43 classes · 32 × 32 RGB · TensorFlow/Keras**")
    st.write("Upload a tightly cropped photograph containing one German traffic sign.")
    st.caption("Educational prototype. Not for driving decisions.")
    st.markdown("[Source & attribution](https://github.com/Kanishkar843/RoadSignNet)")

model_dir = Path(os.environ.get("ROADSIGNNET_MODEL_DIR", "models/roadsignnet"))


@st.cache_resource
def cached_bundle(folder, model_mtime, metadata_mtime):
    return load_bundle(folder)


left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("01 · Your sign")
    upload = st.file_uploader("Select a cropped sign", type=["jpg", "jpeg", "png"])
    st.caption("JPG or PNG · up to 20 MB · up to 25 megapixels. Images are processed in memory.")

with right:
    st.subheader("02 · Prediction")
    if not (model_dir / "model.keras").is_file() or not (model_dir / "metadata.json").is_file():
        st.info("Train RoadSignNet to enable predictions.")
        st.code("python -m roadsignnet.train --data-dir data --output models/roadsignnet")
        st.caption("No pretrained weights are bundled. Setup and dataset instructions are in README.md.")
    elif upload is None:
        st.info("Choose a sign image to see its predicted category and top three alternatives.")
    else:
        try:
            model, metadata = cached_bundle(str(model_dir), (model_dir / "model.keras").stat().st_mtime_ns,
                                            (model_dir / "metadata.json").stat().st_mtime_ns)
            with st.spinner("Reading your sign…"):
                result = predict_image(model, upload.getvalue(), threshold)
            with left:
                st.image(upload.getvalue(), caption=upload.name, width="stretch")
            st.markdown(f"### {result['label']}")
            st.metric("Top model score", f"{result['score']:.1%}")
            if result["score"] < threshold:
                st.warning("Low confidence. Review the image and prediction.")
            st.caption("All predictions need review; unknown objects can also receive high scores.")
            table = pd.DataFrame(result["top_three"])[["label", "score"]].set_index("label")
            st.bar_chart(table, color="#52dbb0", horizontal=True)
            st.download_button("Download prediction", json.dumps(result, indent=2),
                               file_name="roadsignnet_prediction.json", mime="application/json")
        except (ValueError, OSError, KeyError) as exc:
            st.error(str(exc))

st.divider()
st.caption("One cropped sign → RGB preprocessing → compact CNN → 43 class scores. "
           "This application does not detect signs in full road scenes.")
