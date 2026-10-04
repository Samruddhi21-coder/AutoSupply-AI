import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import math
import re
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ============================================================
# AUTO SUPPLY AI
# A clean, data-driven automobile demand forecasting dashboard.
# ============================================================

st.set_page_config(
    page_title="AutoSupply AI | Vehicle Demand Forecasting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
MONTH_MAP = {m: i for i, m in enumerate(MONTHS, 1)}

# Six user-friendly vehicle groups used by the project.
VEHICLE_MAP = {
    "Two Wheeler": ["Two Wheeler(T)", "Two Wheeler(Nt)", "Two Wheeler (Invalid Carriage)"],
    "Three Wheeler": ["Three Wheeler(T)", "Three Wheeler(Nt)"],
    "Passenger Vehicle": ["Light Motor Vehicle", "Four Wheeler (Invalid Carriage)"],
    "Passenger Transport": ["Light Passenger Vehicle", "Medium Passenger Vehicle", "Heavy Passenger Vehicle"],
    "Goods Vehicle": ["Light Goods Vehicle", "Medium Goods Vehicle", "Heavy Goods Vehicle"],
    "Other Motor Vehicle": ["Heavy Motor Vehicle", "Medium Motor Vehicle", "Other Than Mentioned Above"],
}

# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #070a12;
        --panel: #101521;
        --panel2: #141b2b;
        --text: #f5f7ff;
        --muted: #9ba7bd;
        --accent: #7c5cff;
        --accent2: #20d9c2;
        --line: rgba(255,255,255,.09);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(124,92,255,.16), transparent 28%),
            radial-gradient(circle at 90% 12%, rgba(32,217,194,.10), transparent 25%),
            linear-gradient(180deg, #070a12 0%, #090d16 55%, #070a12 100%);
        color: var(--text);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -.02em;
    }

    .eyebrow {
        color: #9da9bf;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: .7rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.8rem, 6vw, 5.6rem);
        line-height: .98;
        letter-spacing: -.065em;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(100deg, #ffffff 5%, #bcb0ff 48%, #73f0df 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        color: #aeb8cb;
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 780px;
        margin-top: 1.4rem;
    }

    .hero-chip {
        display: inline-block;
        border: 1px solid rgba(124,92,255,.35);
        background: rgba(124,92,255,.10);
        color: #c9c0ff;
        border-radius: 999px;
        padding: .42rem .75rem;
        font-size: .78rem;
        margin: .25rem .35rem .25rem 0;
    }

    .feature-card, .info-card, .metric-card, .result-card {
        background: linear-gradient(145deg, rgba(18,24,38,.92), rgba(11,15,25,.92));
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: 0 18px 60px rgba(0,0,0,.22);
    }

    .feature-card {
        padding: 1.35rem;
        min-height: 175px;
    }

    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: .8rem;
    }

    .feature-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: .45rem;
    }

    .feature-text {
        color: #9ba7bd;
        line-height: 1.55;
        font-size: .88rem;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -.03em;
    }

    .metric-card {
        padding: 1.25rem 1.35rem;
        min-height: 145px;
    }

    .metric-label {
        color: #9ba7bd;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 700;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        margin-top: .55rem;
        letter-spacing: -.04em;
    }

    .metric-help {
        color: #7f8ba0;
        font-size: .78rem;
        margin-top: .4rem;
    }

    .result-card {
        padding: 1.7rem;
        border-color: rgba(124,92,255,.28);
        background:
            radial-gradient(circle at 85% 20%, rgba(32,217,194,.09), transparent 28%),
            radial-gradient(circle at 15% 80%, rgba(124,92,255,.10), transparent 32%),
            rgba(13,18,30,.96);
    }

    .result-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4.4rem);
        font-weight: 700;
        letter-spacing: -.06em;
        margin: .25rem 0;
    }

    .result-sub {
        color: #aeb8cb;
        font-size: .95rem;
    }

    .small-muted {
        color: #8793a9;
        font-size: .8rem;
        line-height: 1.55;
    }

    .status-pill {
        display: inline-block;
        padding: .36rem .7rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 700;
        background: rgba(32,217,194,.10);
        color: #74f0df;
        border: 1px solid rgba(32,217,194,.20);
    }

    div[data-testid="stButton"] > button {
        border-radius: 13px;
        border: 1px solid rgba(124,92,255,.45);
        background: linear-gradient(135deg, #7558ff, #4c83ff);
        color: white;
        font-weight: 700;
        min-height: 3rem;
        transition: all .2s ease;
        box-shadow: 0 10px 30px rgba(92,83,255,.20);
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 36px rgba(92,83,255,.30);
        border-color: rgba(255,255,255,.35);
    }

    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stNumberInput"] > div > div {
        background: #0e1420;
        border-radius: 12px;
    }

    .footer-note {
        text-align: center;
        color: #667188;
        font-size: .75rem;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA DISCOVERY
# ============================================================

def _normalise_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def discover_source_csv():
    """Find the raw monthly vehicle dataset by schema, not by hardcoded filename."""
    candidates = []
    for path in BASE_DIR.rglob("*.csv"):
        # Ignore generated exports when possible.
        if path.name.lower() in {"autosupply_inventory_data.csv"}:
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        norm = {_normalise_col(c) for c in cols}
        required = {"date", "state_name", "vehicle_type", "registrations"}
        score = len(required.intersection(norm))
        if score == 4:
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            candidates.append((size, path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


@st.cache_data(show_spinner=False)
def load_raw_data():
    path = discover_source_csv()
    if path is None:
        raise FileNotFoundError(
            "Could not find a CSV containing date, state_name, vehicle_type and registrations."
        )

    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        n = _normalise_col(c)
        if n == "date": rename[c] = "date"
        elif n == "state_name": rename[c] = "state_name"
        elif n == "vehicle_type": rename[c] = "vehicle_type"
        elif n == "registrations": rename[c] = "registrations"
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["registrations"] = pd.to_numeric(df["registrations"], errors="coerce")
    df = df.dropna(subset=["date", "state_name", "vehicle_type", "registrations"])
    df["registrations"] = df["registrations"].clip(lower=0)

    # Map the project's raw Vahan vehicle types into the six dashboard groups.
    reverse_map = {}
    for group, raw_types in VEHICLE_MAP.items():
        for raw in raw_types:
            reverse_map[raw] = group
    df["vehicle_group"] = df["vehicle_type"].map(reverse_map)
    df = df.dropna(subset=["vehicle_group"])

    monthly = (
        df.assign(month_date=df["date"].dt.to_period("M").dt.to_timestamp())
          .groupby(["month_date", "state_name", "vehicle_group"], as_index=False)["registrations"]
          .sum()
          .rename(columns={"month_date": "date", "state_name": "state", "registrations": "demand"})
    )

    # Keep a complete monthly grid for every observed state × vehicle group.
    # Missing combinations are retained only where they actually occur; missing months
    # inside an observed series are filled with zero so lag features are well-defined.
    monthly = monthly.sort_values(["state", "vehicle_group", "date"]).reset_index(drop=True)
    return monthly, path


# ============================================================
# MODEL TRAINING
# ============================================================

def add_time_features(frame):
    out = frame.copy()
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    min_date = out["date"].min()
    out["time_index"] = (
        (out["date"].dt.year - min_date.year) * 12
        + (out["date"].dt.month - min_date.month)
    )
    return out


def make_supervised(monthly):
    pieces = []
    for (state, vehicle), g in monthly.groupby(["state", "vehicle_group"], sort=False):
        g = g.sort_values("date").copy()
        full_dates = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
        g = g.set_index("date").reindex(full_dates)
        g.index.name = "date"
        g["state"] = state
        g["vehicle_group"] = vehicle
        g["demand"] = g["demand"].fillna(0.0)
        g = g.reset_index()
        g = add_time_features(g)
        for lag in [1, 2, 3, 6, 12]:
            g[f"lag_{lag}"] = g["demand"].shift(lag)
        g["roll_3"] = g["demand"].shift(1).rolling(3).mean()
        g["roll_6"] = g["demand"].shift(1).rolling(6).mean()
        g["roll_12"] = g["demand"].shift(1).rolling(12).mean()
        pieces.append(g)

    data = pd.concat(pieces, ignore_index=True)
    data = data.dropna(subset=[
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "roll_3", "roll_6", "roll_12"
    ])
    return data


@st.cache_resource(show_spinner=False)
def train_model(monthly):
    train = make_supervised(monthly)

    feature_cols = [
        "state", "vehicle_group", "year", "month",
        "month_sin", "month_cos", "time_index",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "roll_3", "roll_6", "roll_12",
    ]
    categorical = ["state", "vehicle_group"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    # Random forest is used with lagged time-series features. Recursive forecasting
    # allows the model to produce a different prediction for each requested month/year.
    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    X = train[feature_cols]
    y = np.log1p(train["demand"].clip(lower=0))
    pipe.fit(X, y)
    return pipe, feature_cols


# ============================================================
# FORECAST ENGINE
# ============================================================

def _series_for(monthly, state, vehicle):
    s = monthly[(monthly["state"] == state) & (monthly["vehicle_group"] == vehicle)].copy()
    return s.sort_values("date").reset_index(drop=True)


def _predict_one(pipe, state, vehicle, target_date, history, base_date, feature_cols):
    history = history.copy().sort_values("date")

    # Need at least 12 months for the lagged model.
    if len(history) < 12:
        raise ValueError("Not enough historical observations for this state and vehicle type.")

    current_last = history["date"].max()
    target_date = pd.Timestamp(target_date).to_period("M").to_timestamp()

    # If target is historical, return actual demand separately elsewhere.
    # For future dates, forecast month-by-month recursively.
    if target_date <= current_last:
        row = history.loc[history["date"] == target_date]
        if not row.empty:
            return float(row.iloc[0]["demand"]), history, False

    work = history[["date", "demand"]].copy()
    future_dates = pd.date_range(current_last + pd.offsets.MonthBegin(1), target_date, freq="MS")

    for d in future_dates:
        month = d.month
        year = d.year
        tmp = pd.DataFrame({"date": [d], "state": [state], "vehicle_group": [vehicle]})
        tmp = add_time_features(tmp)

        vals = work.set_index("date")["demand"]
        def val(offset):
            dt = d - pd.DateOffset(months=offset)
            return float(vals.get(dt, np.nan))

        tmp["lag_1"] = val(1)
        tmp["lag_2"] = val(2)
        tmp["lag_3"] = val(3)
        tmp["lag_6"] = val(6)
        tmp["lag_12"] = val(12)

        last12 = [val(i) for i in range(1, 13)]
        last6 = [val(i) for i in range(1, 7)]
        last3 = [val(i) for i in range(1, 4)]
        tmp["roll_3"] = np.nanmean(last3)
        tmp["roll_6"] = np.nanmean(last6)
        tmp["roll_12"] = np.nanmean(last12)

        # Fill any missing lag conservatively with the available recent mean.
        recent = float(work["demand"].tail(12).mean())
        for c in ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll_3", "roll_6", "roll_12"]:
            tmp[c] = tmp[c].fillna(recent)

        pred_log = float(pipe.predict(tmp[feature_cols])[0])
        pred = max(0.0, math.expm1(pred_log))
        work = pd.concat([work, pd.DataFrame({"date": [d], "demand": [pred]})], ignore_index=True)

    final = float(work.loc[work["date"] == target_date, "demand"].iloc[0])
    return final, work, True


def calculate_safety_stock(history, forecast):
    """Data-driven uncertainty buffer based on recent monthly volatility."""
    s = history["demand"].astype(float).clip(lower=0)
    if len(s) >= 13:
        # Compare demand with the same month one year earlier where possible.
        seasonal_error = (s.iloc[12:].to_numpy() - s.iloc[:-12].to_numpy())
        sigma = float(np.std(seasonal_error, ddof=1)) if len(seasonal_error) > 1 else 0.0
    else:
        sigma = float(s.tail(12).std(ddof=1)) if len(s) > 1 else 0.0

    # 90% one-sided service level.
    z = 1.2816
    safety = max(0.0, z * sigma)

    # Avoid a mathematically huge buffer when a very volatile raw series exists.
    safety = min(safety, max(forecast * 0.50, 0.0))
    return safety


def format_number(x):
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:,.0f}"


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf_report(state, vehicle, target_date, demand, safety_stock, recommended,
                     is_future, buffer_pct, actual_demand=None):
    """Create a compact downloadable forecast/result report."""
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=42, bottomMargin=42
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AutoSupplyTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=22, leading=28, textColor=colors.HexColor("#22223b"),
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        "AutoSupplySubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=10, textColor=colors.HexColor("#666b7a"), spaceAfter=18
    )
    body_style = ParagraphStyle(
        "AutoSupplyBody", parent=styles["BodyText"], fontSize=10,
        leading=15, textColor=colors.HexColor("#333333")
    )

    story = [
        Paragraph("AutoSupply AI", title_style),
        Paragraph("Vehicle Demand Forecast & Inventory Planning Report", subtitle_style),
        Paragraph(
            f"<b>Location:</b> {state}<br/>"
            f"<b>Vehicle Type:</b> {vehicle}<br/>"
            f"<b>Target Period:</b> {target_date.strftime('%B %Y')}<br/>"
            f"<b>Status:</b> {'Forecast' if is_future else 'Historical Actual'}",
            body_style
        ),
        Spacer(1, 14),
    ]

    rows = [
        ["Metric", "Value"],
        ["Demand", f"{demand:,.0f} vehicles"],
        ["Safety Stock", f"{safety_stock:,.0f} vehicles"],
        ["Recommended Inventory", f"{recommended:,.0f} vehicles"],
        ["Inventory Buffer", f"{buffer_pct:.2f}%"],
    ]
    if actual_demand is not None:
        rows.insert(2, ["Historical Actual", f"{actual_demand:,.0f} vehicles"])

    table = Table(rows, colWidths=[240, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6c5ce7")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d9dce5")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f7f8fb")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f8fb")]),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story.extend([table, Spacer(1, 16)])

    story.append(Paragraph(
        f"<b>Planning logic:</b> Recommended Inventory = Demand + Safety Stock. "
        f"The inventory buffer is the safety stock expressed as a percentage of demand: "
        f"{safety_stock:,.0f} / {demand:,.0f} = {buffer_pct:.2f}%.",
        body_style
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "AutoSupply AI uses the project's historical monthly vehicle-registration data and its trained forecasting model. "
        "A forecast is an estimate and should be validated against operational constraints before procurement decisions.",
        body_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# LOAD DATA + MODEL
# ============================================================

try:
    monthly, source_path = load_raw_data()
    model, FEATURE_COLS = train_model(monthly)
except Exception as exc:
    st.error("AutoSupply AI could not initialize the forecasting engine.")
    st.code(str(exc))
    st.stop()

states = sorted(monthly["state"].dropna().unique().tolist())
vehicles = [v for v in VEHICLE_MAP.keys() if v in set(monthly["vehicle_group"].unique())]
min_year = int(monthly["date"].dt.year.min())
max_hist_year = int(monthly["date"].dt.year.max())
max_forecast_year = max_hist_year + 6


# ============================================================
# LANDING PAGE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown(
        "<div class='brand'>🚗 AUTOSUPPLY AI</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("<div class='eyebrow'>AI • AUTOMOTIVE • FORECASTING</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-title'>Know the demand.<br>Plan the vehicles.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='hero-copy'>
        AutoSupply AI turns historical vehicle-registration data into a simple planning answer:
        <b>how many vehicles could be needed for a chosen state, vehicle type, month and year?</b>
        It then adds a data-driven safety buffer so inventory decisions are easier to understand.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='hero-chip'>Monthly forecasting</span><span class='hero-chip'>State-aware</span><span class='hero-chip'>Vehicle-aware</span><span class='hero-chip'>Inventory planning</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    _, btn_col, _ = st.columns([1, 1.15, 1])
    with btn_col:
        if st.button("✨  LET'S EXPLORE", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

    st.write("")
    st.markdown("<div class='section-title'>What problem does it solve?</div>", unsafe_allow_html=True)
    st.write("")

    cards = [
        ("🎯", "Uncertain demand", "Estimate future vehicle demand before deciding how many vehicles need to be planned."),
        ("📍", "Different markets", "Demand is not identical across Indian states. The selected location is part of the forecast."),
        ("📅", "Time changes demand", "Month and year are real model inputs, so the forecast changes with the requested period."),
        ("🛡️", "Demand uncertainty", "A statistical safety buffer helps protect the plan from normal demand variability."),
    ]
    cols = st.columns(4)
    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon'>{icon}</div><div class='feature-title'>{title}</div><div class='feature-text'>{text}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("<div class='section-title'>How AutoSupply AI works</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown(
        """
        <div class='info-card' style='padding:1.5rem;'>
        <div style='display:flex;gap:12px;flex-wrap:wrap;align-items:center;'>
          <span class='hero-chip'>01 Historical data</span>
          <span style='color:#667188'>→</span>
          <span class='hero-chip'>02 Time-series features</span>
          <span style='color:#667188'>→</span>
          <span class='hero-chip'>03 Trained forecasting model</span>
          <span style='color:#667188'>→</span>
          <span class='hero-chip'>04 Safety stock</span>
          <span style='color:#667188'>→</span>
          <span class='hero-chip'>05 Planning recommendation</span>
        </div>
        <div class='small-muted' style='margin-top:1rem;'>
        The forecasting engine is trained from the project's historical monthly vehicle-registration data.
        The dashboard does not use a hard-coded demand number for different selections.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='footer-note'>Historical data available from {min_year} to {max_hist_year} • {len(states)} states • {len(vehicles)} vehicle groups</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# DASHBOARD PAGE
# ============================================================

# Top navigation
nav1, nav2, nav3 = st.columns([2.5, 1, 1])
with nav1:
    st.markdown("<div class='brand'>🚗 AutoSupply AI</div>", unsafe_allow_html=True)
with nav3:
    if st.button("← Back to overview", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
st.markdown("<div class='eyebrow'>FORECAST WORKSPACE</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title' style='font-size:2.2rem;'>What will the market need?</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-muted'>Choose a location, vehicle type and target month/year. AutoSupply AI will generate a model-based demand estimate and inventory buffer.</div>",
    unsafe_allow_html=True,
)
st.write("")

# Input panel
with st.container(border=True):
    st.markdown("### 🎛️ Forecast inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_state = st.selectbox("📍 Location / State", states, key="state_input")
    with c2:
        selected_vehicle = st.selectbox("🚘 Vehicle Type", vehicles, key="vehicle_input")
    with c3:
        selected_month = st.selectbox("📅 Month", MONTHS, index=5, key="month_input")
    with c4:
        selected_year = st.number_input(
            "🗓️ Year",
            min_value=min_year,
            max_value=max_forecast_year,
            value=min(max_hist_year + 1, max_forecast_year),
            step=1,
            key="year_input",
        )

    target_date = pd.Timestamp(year=int(selected_year), month=MONTH_MAP[selected_month], day=1)
    series = _series_for(monthly, selected_state, selected_vehicle)

    if len(series) < 12:
        st.warning("This state + vehicle combination has less than 12 months of history. Choose another combination for a reliable forecast.")
        st.stop()

    generate = st.button("✨  GENERATE FORECAST", use_container_width=True)

# Persist selected result after button click.
selection_key = f"{selected_state}|{selected_vehicle}|{target_date.strftime('%Y-%m')}"
if generate or st.session_state.get("last_selection") != selection_key:
    try:
        with st.spinner("Analyzing historical demand and generating forecast…"):
            forecast, forecast_history, is_future = _predict_one(
                model,
                selected_state,
                selected_vehicle,
                target_date,
                series,
                monthly["date"].min(),
                FEATURE_COLS,
            )
            safety = calculate_safety_stock(series, forecast)
            recommended = forecast + safety
            st.session_state.last_selection = selection_key
            st.session_state.result = {
                "forecast": forecast,
                "safety": safety,
                "recommended": recommended,
                "is_future": is_future,
                "series": series.copy(),
                "forecast_history": forecast_history.copy(),
                "target_date": target_date,
            }
    except Exception as exc:
        st.error(f"Forecast could not be generated: {exc}")
        st.stop()

result = st.session_state.get("result")
if not result:
    st.info("Choose your inputs and click Generate Forecast.")
    st.stop()

forecast = result["forecast"]
safety = result["safety"]
recommended = result["recommended"]
series = result["series"]
forecast_history = result["forecast_history"]
target_date = result["target_date"]
is_future = result["is_future"]

buffer_pct = (safety / forecast * 100) if forecast > 0 else 0
coverage_pct = (recommended / forecast * 100) if forecast > 0 else 0

# Status
status_text = "FORECAST" if is_future else "HISTORICAL ACTUAL"
st.markdown(
    f"<span class='status-pill'>{status_text}</span> <span class='small-muted' style='margin-left:.5rem;'>{selected_state} · {selected_vehicle} · {target_date.strftime('%B %Y')}</span>",
    unsafe_allow_html=True,
)
st.write("")

# Main result card
st.markdown(
    f"""
    <div class='result-card'>
      <div class='metric-label'>AI planning recommendation</div>
      <div class='result-number'>{format_number(recommended)} vehicles</div>
      <div class='result-sub'>Recommended planning quantity for <b>{selected_vehicle}</b> in <b>{selected_state}</b> during <b>{target_date.strftime('%B %Y')}</b>.</div>
      <div class='small-muted' style='margin-top:1rem;'>
        Recommended quantity = predicted demand + safety stock.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Metrics
m1, m2, m3, m4 = st.columns(4)
demand_label = "🎯 Predicted Demand" if is_future else "📌 Actual Demand"
demand_help = "Expected vehicles needed from the model" if is_future else "Recorded vehicles in the selected historical month"
metrics = [
    (demand_label, format_number(forecast), demand_help),
    ("🛡️ Safety Stock", format_number(safety), "Extra buffer for uncertainty"),
    ("📦 Recommended Inventory", format_number(recommended), "Demand + safety stock"),
    ("📊 Inventory Buffer", f"{buffer_pct:.2f}%", "Extra inventory above demand"),
]
for col, (label, value, help_text) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-help'>{help_text}</div></div>",
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    f"<div class='small-muted'>Inventory buffer = <b>{buffer_pct:.2f}%</b> extra above predicted demand. Recommended inventory is <b>{coverage_pct:.2f}%</b> of predicted demand.</div>",
    unsafe_allow_html=True,
)

# Historical vs target explanation
if is_future:
    st.markdown(
        f"<div class='info-card' style='padding:1rem 1.25rem;'>📈 <b>{target_date.strftime('%B %Y')}</b> is beyond the historical data available through <b>{series['date'].max().strftime('%B %Y')}</b>. The displayed demand is a model forecast, not an actual registration count.</div>",
        unsafe_allow_html=True,
    )
else:
    actual = float(series.loc[series["date"] == target_date, "demand"].iloc[0])
    st.markdown(
        f"<div class='info-card' style='padding:1rem 1.25rem;'>✅ Actual historical demand for <b>{target_date.strftime('%B %Y')}</b> is <b>{actual:,.0f}</b> vehicles. No future prediction is being presented as actual data.</div>",
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("<div class='section-title'>Demand intelligence</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-muted'>A few focused visuals are shown below. They are designed to explain the forecast rather than overwhelm you with charts.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================
# CHART 1: HISTORICAL + FORECAST
# ============================================================

hist = series[["date", "demand"]].copy()
chart_end = max(target_date, hist["date"].max())
chart_start = max(hist["date"].min(), chart_end - pd.DateOffset(months=30))
chart_hist = hist[(hist["date"] >= chart_start) & (hist["date"] <= hist["date"].max())]

# Build a small forecast path from the last historical point to target.
plot_forecast = forecast_history.copy()
plot_forecast["date"] = pd.to_datetime(plot_forecast["date"])
plot_forecast = plot_forecast[plot_forecast["date"] > hist["date"].max()].copy()
plot_forecast = plot_forecast[plot_forecast["date"] <= target_date].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=chart_hist["date"], y=chart_hist["demand"],
    mode="lines", name="Historical demand",
    line=dict(color="#8b7cff", width=3),
    fill="tozeroy", fillcolor="rgba(124,92,255,.08)",
))
if not plot_forecast.empty:
    bridge = pd.DataFrame({
        "date": [hist["date"].max()],
        "demand": [hist["demand"].iloc[-1]],
    })
    pf = pd.concat([bridge, plot_forecast[["date", "demand"]]], ignore_index=True)
    fig.add_trace(go.Scatter(
        x=pf["date"], y=pf["demand"],
        mode="lines+markers", name="Model forecast",
        line=dict(color="#20d9c2", width=3, dash="dot"),
        marker=dict(size=6),
    ))

fig.add_trace(go.Scatter(
    x=[target_date], y=[forecast],
    mode="markers", name="Selected period",
    marker=dict(size=13, color="#ffffff", line=dict(color="#7c5cff", width=4)),
))
fig.update_layout(
    title=f"{selected_state} · {selected_vehicle} demand history",
    height=430,
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd3e3"),
    legend=dict(orientation="h", y=1.02, x=0),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", title="Vehicles"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# CHART 2 + 3
# ============================================================

left, right = st.columns(2)

with left:
    seasonal = (
        series.assign(month_num=series["date"].dt.month)
        .groupby("month_num", as_index=False)["demand"].mean()
    )
    seasonal["month"] = seasonal["month_num"].map(lambda x: MONTHS[x - 1])
    seasonal = seasonal.sort_values("month_num")

    fig_season = go.Figure()
    fig_season.add_trace(go.Bar(
        x=seasonal["month"],
        y=seasonal["demand"],
        marker_color="#7c5cff",
        hovertemplate="%{x}<br>Average demand: %{y:,.0f}<extra></extra>",
    ))
    fig_season.update_layout(
        title="Typical monthly demand pattern",
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd3e3"),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.06)", title="Average vehicles"),
    )
    st.plotly_chart(fig_season, use_container_width=True, config={"displayModeBar": False})

with right:
    inv = pd.DataFrame({
        "Component": ["Predicted demand", "Safety stock"],
        "Vehicles": [forecast, safety],
    })
    fig_inv = go.Figure(go.Bar(
        x=inv["Component"], y=inv["Vehicles"],
        marker_color=["#7c5cff", "#20d9c2"],
        text=[format_number(forecast), format_number(safety)],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>%{y:,.0f} vehicles<extra></extra>",
    ))
    fig_inv.update_layout(
        title="How the recommendation is built",
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd3e3"),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.06)", title="Vehicles"),
        showlegend=False,
    )
    st.plotly_chart(fig_inv, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# SIMPLE EXPLANATION
# ============================================================

st.markdown("<div class='section-title'>🧠 What this result means</div>", unsafe_allow_html=True)
st.write("")

st.markdown(
    f"""
    <div class='info-card' style='padding:1.4rem;'>
      <div style='font-size:1.05rem;line-height:1.75;color:#d9dfeb;'>
      For <b>{selected_state}</b> and <b>{selected_vehicle}</b> in <b>{target_date.strftime('%B %Y')}</b>,
      AutoSupply AI estimates demand of <b>{forecast:,.0f} vehicles</b>.
      Based on the historical variability of this demand series, the system adds approximately
      <b>{safety:,.0f} vehicles</b> as a safety buffer.
      Therefore, the planning recommendation is approximately <b>{recommended:,.0f} vehicles</b>.
      </div>
      <div class='small-muted' style='margin-top:1rem;'>
      Formula: Recommended Inventory = Predicted Demand + Safety Stock.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.write("")
st.markdown("### 📄 Download this result", unsafe_allow_html=True)

actual_for_pdf = None
if not is_future:
    actual_for_pdf = float(series.loc[series["date"] == target_date, "demand"].iloc[0])

pdf_bytes = build_pdf_report(
    selected_state, selected_vehicle, target_date,
    forecast, safety, recommended, is_future, buffer_pct, actual_for_pdf
)

if pdf_bytes is not None:
    st.download_button(
        "⬇️ Download Forecast Report (PDF)",
        data=pdf_bytes,
        file_name=f"AutoSupply_AI_{selected_state}_{selected_vehicle}_{target_date.strftime('%Y_%m')}.pdf".replace(" ", "_"),
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.info("PDF export needs the reportlab package. Install it with: pip install reportlab")

st.markdown(
    f"<div class='footer-note'>AutoSupply AI • Model trained on the project's historical monthly vehicle-registration data • PDF reports available • Source: {source_path.name}</div>",
    unsafe_allow_html=True,
)
