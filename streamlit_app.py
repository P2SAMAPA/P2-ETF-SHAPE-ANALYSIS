import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Shape Theoretic Analysis", layout="wide")
st.title("📐 Morphological Shape Analysis – Trading Signals")
st.caption("Kendall's shape space | Procrustes distance | V = BUY, U = HOLD, L = SELL")

# Custom CSS
st.markdown("""
<style>
    .buy { background-color: #c6f7d0; padding: 0.2rem; border-radius: 0.5rem; text-align: center; color: #0e6b0e; font-weight: bold; }
    .hold { background-color: #fff3c4; padding: 0.2rem; border-radius: 0.5rem; text-align: center; color: #b45f06; }
    .sell { background-color: #fdd; padding: 0.2rem; border-radius: 0.5rem; text-align: center; color: #a00; }
    .info-text { font-size: 0.9rem; color: #555; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

def shape_to_score(shape):
    mapping = {"V": 1, "U": 0, "L": -1}
    return mapping.get(shape, 0)

def shape_to_action(shape):
    mapping = {"V": "BUY", "U": "HOLD", "L": "SELL"}
    return mapping.get(shape, "HOLD")

def color_action(val):
    if val == "BUY":
        return 'background-color: #c6f7d0; color: #0e6b0e; font-weight: bold;'
    elif val == "HOLD":
        return 'background-color: #fff3c4; color: #b45f06;'
    elif val == "SELL":
        return 'background-color: #fdd; color: #a00;'
    return ''

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'shape_analysis' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No shape analysis results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Procrustes alignment, k‑medoids clustering on recovery segments.")
st.sidebar.write("**Normalised time:** 0 = trough, 1 = recovery peak. Shape shows *how* price recovered, not how long.")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

# ================== RECOMMENDATION SECTION ==================
st.header("🎯 Top ETFs to Buy (based on shape confidence and Procrustes distance)")

all_recommendations = []  # list of (universe, ticker, score, shape, confidence, distance)
for universe_name, uni_data in universes.items():
    for ticker, info in uni_data.items():
        shape = info.get("current_shape", "?")
        score = shape_to_score(shape)
        confidence = info.get("confidence", 0.0)
        # Composite score: shape_score * confidence (positive for V, zero for U, negative for L)
        composite = score * confidence
        # Also compute a "buy_score" that is positive only for V: shape_score * confidence
        all_recommendations.append({
            "Universe": universe_name,
            "Ticker": ticker,
            "Shape": shape,
            "Action": shape_to_action(shape),
            "Confidence": f"{confidence*100:.1f}%",
            "Procrustes Dist": f"{info['procrustes_distance']:.3f}",
            "Composite Score": composite,
            "confidence_raw": confidence,
            "distance": info['procrustes_distance']
        })
# Sort by composite descending (V with high confidence at top)
df_rec = pd.DataFrame(all_recommendations)
df_rec = df_rec.sort_values("Composite Score", ascending=False)
top3 = df_rec.head(3)
# Display as hero cards
col1, col2, col3 = st.columns(3)
for i, row in top3.iterrows():
    with eval(f"col{i+1}"):
        st.markdown(f"##### {row['Universe']} – {row['Ticker']}")
        st.markdown(f"**Action:** {row['Action']}")
        st.markdown(f"**Confidence:** {row['Confidence']}")
        st.markdown(f"**Procrustes distance:** {row['Procrustes Dist']}")
        st.markdown(f"**Shape:** {row['Shape']}")
        if row['Action'] == "BUY":
            st.markdown('<div class="buy">BUY SIGNAL</div>', unsafe_allow_html=True)
        elif row['Action'] == "HOLD":
            st.markdown('<div class="hold">HOLD SIGNAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sell">SELL SIGNAL</div>', unsafe_allow_html=True)
st.divider()

# ================== UNIVERSE SELECTION ==================
universe_names = list(universes.keys())
selected = st.selectbox("Select Universe to view details", universe_names)

if selected:
    uni_data = universes[selected]
    if not uni_data:
        st.write("No data for this universe.")
        st.stop()
    
    rows = []
    for ticker, info in uni_data.items():
        shape = info.get("current_shape", "?")
        action = shape_to_action(shape)
        rows.append({
            "ETF": ticker,
            "Current Shape": shape,
            "Action": action,
            "Confidence": f"{info.get('confidence',0)*100:.1f}%",
            "Recoveries": info.get("num_recoveries", 0),
            "Procrustes Dist": f"{info.get('procrustes_distance',0):.3f}"
        })
    df = pd.DataFrame(rows)
    styled_df = df.style.map(color_action, subset=['Action'])
    st.subheader(f"📊 Trading Signals – {selected}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with st.expander("📘 What does 'Normalised time' mean?"):
        st.markdown("""
        - **Normalised time** scales each recovery to start at 0 (trough) and end at 1 (peak).  
        - This allows comparison of different‑duration recoveries – a 10‑day V‑shape and a 100‑day V‑shape look the same on this axis.  
        - **Normalised price** scales from 0 (trough) to 1 (peak).  
        - The shape tells you *how* the price recovered, not the calendar length.
        """)

    ticker = st.selectbox("Select ETF for details", df["ETF"].tolist())
    if ticker:
        info = uni_data[ticker]
        shape = info['current_shape']
        action = shape_to_action(shape)
        st.markdown(f"### {ticker} – {shape} shape → **{action}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{info['confidence']*100:.1f}%")
        col2.metric("Total recoveries", info['num_recoveries'])
        col3.metric("Procrustes distance", f"{info['procrustes_distance']:.4f}")

        st.markdown("**Cluster distribution (historical recoveries):**")
        cluster_df = pd.DataFrame([
            {"Shape Type": name, "Count": info['cluster_distribution'].get(str(k), 0)}
            for k, name in info['cluster_names'].items()
        ])
        st.bar_chart(cluster_df.set_index("Shape Type"))

        last_seg = info.get("last_recovery_normalized")
        if last_seg and len(last_seg) > 0:
            last_seg = np.array(last_seg)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=last_seg[:,0], y=last_seg[:,1], mode='lines+markers',
                                     name=f"{ticker} - last recovery", line=dict(width=3),
                                     marker=dict(size=4)))
            fig.update_layout(title="Normalised recovery shape (trough → peak)",
                              xaxis_title="Normalised time (0 = trough, 1 = peak)",
                              yaxis_title="Normalised price (0 = trough, 1 = peak)",
                              height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shape data available for this ETF.")

st.caption("Top recommendation uses composite score = shape_score (V=1, U=0, L=-1) × confidence. Higher is better for buying. Data from " + OUTPUT_REPO)
