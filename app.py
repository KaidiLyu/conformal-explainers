"""Read saved experiment artifacts; changing a selection never recalibrates on test data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bamboo Steamer · Conformal explanations", layout="wide")
st.title("Conformal image explanations")
st.caption("Bamboo Steamer · CV reproduction project")
folder = Path(st.sidebar.text_input("Results directory", "outputs/imagenette-v1"))
if not (folder / "summary.csv").is_file():
    st.info(
        "Run an evaluation first, or use outputs/smoke after running conformal-cv smoke."
    )
    st.stop()
metadata = json.loads((folder / "metadata.json").read_text())
if metadata.get("synthetic"):
    st.warning(
        "Synthetic smoke test — these are software checks, not Imagenette research results."
    )
if metadata.get("status") != "complete":
    st.warning("This run is incomplete; only completed results are shown.")
st.caption(
    "Confidence is a dataset-level target. Agreement on one image is not a reliability guarantee."
)
df = pd.read_csv(folder / "summary.csv")
method = st.sidebar.selectbox("Explainer", df.method.unique())
variant = st.sidebar.selectbox("Calibration variant", df.variant.unique())
confidence = st.sidebar.selectbox("Target confidence", sorted(df.confidence.unique()))
selected = df[
    (df.method == method) & (df.variant == variant) & (df.confidence == confidence)
]
st.dataframe(selected, hide_index=True)
examples = sorted(folder.glob(f"{method}_{variant}_{confidence}_*.npz"))
if examples:
    path = st.selectbox("Example", examples, format_func=lambda p: p.stem)
    with np.load(path, allow_pickle=False) as a:
        cols = st.columns(4)
        cols[0].image(a["image"], caption="Original image")
        attr = a["attribution"]
        attr = (attr - attr.min()) / max(float(np.ptp(attr)), 1e-12)
        cols[1].image(attr, caption="Attribution (display normalized)")
        cols[2].image(a["mask"].astype(float), caption="Retained region")
        cols[3].image(a["masked_image"], caption="Masked image")
        st.write(
            f"Original class: {int(a['prediction'])} · Masked class: {int(a['masked_prediction'])} · Retained: {float(a['retained_fraction']):.1%}"
        )
st.subheader("All results")
st.dataframe(df, hide_index=True)
st.download_button(
    "Download summary CSV",
    (folder / "summary.csv").read_bytes(),
    "summary.csv",
    "text/csv",
)
