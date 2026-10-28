"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Modules
  Tab 1  Site Safety & PPE Audit            (Epic 2)
  Tab 2  Crack Diagnostics & Integrity       (Epic 2)
  Tab 3  Planning & Real Estate Valuation    (Epic 3)
  Tab 4  Daily Site Assistant & Weather      (Epic 5)

Run locally:   streamlit run app.py
Deploy:        push app.py + requirements.txt to GitHub -> Streamlit Community Cloud
No database, no local files. All state lives in st.session_state.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.parse
from datetime import datetime

import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {
    background: linear-gradient(120deg, #0f2b46 0%, #1c5d99 60%, #f59e0b 140%);
    color: #fff; padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
    box-shadow: 0 6px 20px rgba(15,43,70,.25);
}
.hero h1 {margin: 0; font-size: 1.9rem; letter-spacing: .2px; color: #fff;}
.hero p {margin: .35rem 0 0; opacity: .9; font-size: .97rem;}
.card {
    background: #ffffff; border: 1px solid #e5e9f0; border-radius: 14px;
    padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,.05); height: 100%;
}
.card .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; color: #6b7686;}
.card .value {font-size: 1.75rem; font-weight: 700; color: #0f2b46; line-height: 1.25;}
.card .sub {font-size: .8rem; color: #6b7686;}
.pill {
    display: inline-block; padding: .2rem .75rem; border-radius: 999px;
    font-size: .78rem; font-weight: 600; letter-spacing: .02em; color: #fff;
}
.pill.green {background: #16a34a;} .pill.amber {background: #d97706;}
.pill.red {background: #dc2626;}   .pill.grey {background: #64748b;}
.pill.crit {background: #7f1d1d;}  .pill.blue {background: #2563eb;}
.alert {border-left: 5px solid; padding: .8rem 1rem; border-radius: 8px; margin-bottom: .6rem; background: #f8fafc; color: #1e293b;}
.alert.high {border-color: #dc2626; background: #fef2f2;}
.alert.med {border-color: #d97706; background: #fffbeb;}
.alert.low {border-color: #16a34a; background: #f0fdf4;}
.demo-banner {background:#fff7ed; border:1px solid #fdba74; color:#9a3412; padding:.55rem .9rem; border-radius:10px; font-size:.88rem; margin-bottom:.8rem;}
.small-note {font-size: .78rem; color: #6b7686;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_IMAGE_SIDE = 1600
SEVERITY_PILL = {"Low": "green", "Medium": "amber", "High": "red", "Critical": "crit"}
SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def pill(text: str, color: str) -> str:
    return f'<span class="pill {color}">{text}</span>'


def format_inr(amount: float) -> str:
    """Format INR in Lakhs / Crores."""
    if amount >= 1e7:
        return f"₹ {amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹ {amount / 1e5:,.2f} L"
    return f"₹ {amount:,.0f}"


def load_image(uploaded) -> Image.Image:
    """Open, normalise and downscale an uploaded image."""
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def seed_from(uploaded) -> int:
    return int(hashlib.md5(uploaded.getvalue()).hexdigest()[:8], 16)


def extract_json(text: str) -> dict:
    """Robustly parse JSON from model output (handles ``` fences / preamble)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def get_secret_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Gemini service layer
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    from google import genai  # lazy import so demo mode works without the SDK

    return genai.Client(api_key=api_key)


def gemini_json(api_key: str, model: str, prompt: str, image: Image.Image) -> dict:
    """Multimodal call that returns parsed JSON."""
    from google.genai import types

    client = get_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.2
        ),
    )
    return extract_json(response.text)


ASSISTANT_SYSTEM = (
    "You are SiteMentor, a senior construction engineer assistant for Indian projects. "
    "Ground answers in IS 456 (RCC), IS 800 (steel), IS 4326/1893 (seismic), IS 3764 "
    "(excavation safety), IS 7969 (site safety), the BOCW Act and OSHA 1926. "
    "Be concise, practical, use bullet points and numeric values where relevant, and "
    "explicitly flag when a licensed structural engineer must be consulted. "
    "Never invent code clause numbers you are unsure of."
)


def gemini_chat(api_key: str, model: str, history: list[dict], weather: str) -> str:
    from google.genai import types

    client = get_client(api_key)
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=f"{ASSISTANT_SYSTEM} Current site weather: {weather}.",
            temperature=0.4,
        ),
    )
    return response.text


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
PPE_PROMPT = """You are a certified construction safety officer (OSHA 1926 / IS 7969 / BOCW Act)
and structural reviewer (IS 456:2000 for RCC practices). Analyse the attached construction site photo.

Assess PPE compliance for every visible worker: hard helmet, high-visibility vest,
steel-toe safety boots, safety harness (where work at height is visible).
Also identify site hazards (fall protection, scaffolding, formwork/shuttering, exposed rebar,
electrical, housekeeping, excavation, material stacking, concrete practice per IS 456).

Return ONLY valid JSON, no markdown, exactly this schema:
{
  "safety_score": <integer 0-100>,
  "overall_status": "Compliant" | "Needs Improvement" | "Non-Compliant",
  "workers_detected": <integer>,
  "summary": "<2-3 sentence assessment>",
  "ppe_detection": [
    {"item": "Helmet", "status": "Detected" | "Missing" | "Not Visible", "compliance_pct": <0-100>, "note": "<short>"},
    {"item": "High-Vis Vest", ...},
    {"item": "Steel-Toe Boots", ...},
    {"item": "Safety Harness", ...}
  ],
  "hazards": [
    {"hazard": "<description>", "severity": "Low" | "Medium" | "High" | "Critical",
     "standard": "<OSHA / IS reference>", "remedy": "<corrective action>"}
  ]
}
If the image is not a construction site, set safety_score to 0, overall_status to "Non-Compliant"
and explain in summary."""

CRACK_PROMPT = """You are a senior structural engineer (IS 456:2000, IS 13935 retrofitting, IS 15988).
Analyse the attached concrete / masonry surface image for defects.

Classify into exactly one defect_type:
"Hairline", "Settlement", "Structural Shear", "Thermal Expansion", "Spalling".
Estimate crack width in mm using visible scale cues (state assumptions).
IS 456 cl. 35.3.2 guides: ~0.3 mm general limit, ~0.2 mm for aggressive exposure.

Return ONLY valid JSON, no markdown, exactly this schema:
{
  "defect_type": "<one of the five>",
  "confidence_pct": <0-100>,
  "estimated_crack_width_mm": <number>,
  "crack_pattern": "<vertical / diagonal / horizontal / map / stair-step ...>",
  "structural_risk": "Low" | "Medium" | "High" | "Critical",
  "urgency": "Monitor" | "Within 3 Months" | "Within 2 Weeks" | "Immediate",
  "root_causes": ["..."],
  "recommended_repairs": [{"method": "<e.g. Epoxy injection>", "detail": "<specification>"}],
  "monitoring_plan": "<tell-tales / crack gauges / interval>",
  "engineer_note": "<assumptions and when to call a licensed structural engineer>"
}
If the image does not show a concrete or masonry surface, say so in engineer_note and use Low risk."""

# --------------------------------------------------------------------------- #
# Demo-mode generators (deterministic per image)
# --------------------------------------------------------------------------- #
def demo_ppe(seed: int) -> dict:
    score = 52 + seed % 38
    status = "Compliant" if score >= 85 else "Needs Improvement" if score >= 65 else "Non-Compliant"
    helmet = 70 + seed % 30
    vest = 55 + (seed >> 3) % 45
    boots = 60 + (seed >> 5) % 40
    harness = 20 + (seed >> 7) % 60

    def st_(p):
        return "Detected" if p >= 80 else "Missing" if p < 50 else "Detected"

    return {
        "safety_score": score,
        "overall_status": status,
        "workers_detected": 2 + seed % 5,
        "summary": (
            "DEMO RESULT: Most workers wear head protection, but high-visibility and fall-arrest "
            "compliance is inconsistent near the working edge. Housekeeping and edge protection "
            "need immediate attention."
        ),
        "ppe_detection": [
            {"item": "Helmet", "status": st_(helmet), "compliance_pct": helmet, "note": "Chin straps not secured on some workers."},
            {"item": "High-Vis Vest", "status": st_(vest), "compliance_pct": vest, "note": "Vests faded / not reflective."},
            {"item": "Steel-Toe Boots", "status": st_(boots), "compliance_pct": boots, "note": "Footwear partially obscured."},
            {"item": "Safety Harness", "status": "Missing" if harness < 50 else "Detected", "compliance_pct": harness, "note": "Work at height without visible anchorage."},
        ],
        "hazards": [
            {"hazard": "Unprotected slab edge / missing guard rails", "severity": "Critical", "standard": "OSHA 1926.502; IS 7969", "remedy": "Install 1.0 m top rail, mid rail and toe board; enforce 100% tie-off."},
            {"hazard": "Exposed starter bars without caps", "severity": "High", "standard": "OSHA 1926.701(b)", "remedy": "Fit mushroom caps / bend over protruding rebar ends."},
            {"hazard": "Loose material stacked near edge", "severity": "Medium", "standard": "IS 7969", "remedy": "Relocate stacks >2 m from edge; limit stack height to 1.5 m."},
            {"hazard": "Poor housekeeping / trip hazards", "severity": "Low", "standard": "OSHA 1926.25", "remedy": "Daily housekeeping shift; designate waste collection points."},
        ],
    }


def demo_crack(seed: int) -> dict:
    kinds = [
        ("Hairline", 0.15, "Low", "Monitor",
         ["Plastic shrinkage during early curing", "Inadequate curing / rapid moisture loss"],
         [{"method": "Surface sealing", "detail": "Apply polymer-modified cementitious slurry or elastomeric coating after cleaning."}]),
        ("Settlement", 2.5, "High", "Within 2 Weeks",
         ["Differential foundation settlement", "Poor soil bearing or leaking drainage"],
         [{"method": "Underpinning / grouting", "detail": "Soil investigation, then chemical or cement grouting beneath footing."},
          {"method": "Structural re-grouting", "detail": "Epoxy-grout cracks after movement has stabilised."}]),
        ("Structural Shear", 3.2, "Critical", "Immediate",
         ["Inadequate shear reinforcement / stirrup spacing", "Overloading beyond design capacity"],
         [{"method": "Propping and epoxy injection", "detail": "Prop member, inject low-viscosity epoxy (IS 13935), add FRP / steel jacketing."}]),
        ("Thermal Expansion", 0.6, "Medium", "Within 3 Months",
         ["Absence of expansion joints", "Daily thermal cycling of exposed slab"],
         [{"method": "Joint retrofitting", "detail": "Cut movement joints and fill with polyurethane sealant."}]),
        ("Spalling", 1.2, "High", "Within 2 Weeks",
         ["Reinforcement corrosion from chloride / carbonation", "Insufficient concrete cover"],
         [{"method": "Patch repair", "detail": "Remove delaminated concrete, clean/treat rebar with anti-corrosive primer, apply polymer-modified repair mortar."}]),
    ]
    k = kinds[seed % len(kinds)]
    return {
        "defect_type": k[0],
        "confidence_pct": 72 + seed % 20,
        "estimated_crack_width_mm": round(k[1] + (seed % 7) * 0.05, 2),
        "crack_pattern": ["Vertical", "Diagonal", "Stair-step", "Horizontal", "Map-pattern"][seed % 5],
        "structural_risk": k[2],
        "urgency": k[3],
        "root_causes": k[4],
        "recommended_repairs": k[5],
        "monitoring_plan": "Install glass tell-tales / crack gauges; record width weekly for 8 weeks, then monthly.",
        "engineer_note": "DEMO RESULT. Width estimated without scale reference. Validate on site with a crack comparator; consult a licensed structural engineer for any structural class defect.",
    }


DEMO_CHAT_RULES = [
    (("curing", "cure"), "**Concrete curing (IS 456 cl. 13.5):**\n- Start curing within 24 h; keep moist ≥7 days (OPC) or ≥10 days (blended cements / hot weather).\n- Use wet hessian, ponding or curing compound.\n- In temperatures >40 °C pour at night and use chilled water / retarders."),
    (("cover", "rebar", "reinforcement"), "**Nominal cover (IS 456 Table 16):** Mild 20 mm, Moderate 30 mm, Severe 45 mm, Very severe 50 mm, Extreme 75 mm (columns min 40 mm). Use cover blocks every 1 m."),
    (("slump", "workability"), "**Typical slump (IS 456 cl. 7.1):** Beams/slabs 25–75 mm, columns 75–100 mm, pumped concrete 100–150 mm. Always verify with a slump cone at delivery."),
    (("grade", "mix", "m25", "m20"), "**Mix guidance:** M20 min for RCC (IS 456 Table 5), M25 for moderate exposure, M30+ for severe. Maintain w/c ≤0.50 for moderate exposure."),
    (("crane", "wind"), "**Crane safety:** Suspend lifts at sustained wind ≥ 36 km/h (≈10 m/s) or per manufacturer. Lower the hook, slew-brake free, secure loads."),
    (("scaffold",), "**Scaffolding:** Inspect before each shift, base plates/sole boards, ties every 4 m vertically, full decking and guard rails, no overloading."),
    (("steel", "kg"), "**Rule of thumb:** Residential RCC ≈ 3.5–4.5 kg steel per sq ft built-up; commercial ≈ 4.5–5.5. Use the Planning tab for a project-specific estimate."),
]
DEMO_CHAT_DEFAULT = (
    "I'm running in **demonstration mode** (no API key), so I answer from a small built-in knowledge base. "
    "Try asking about *curing*, *cover*, *slump*, *mix grades*, *crane wind limits* or *scaffolding*. "
    "Add a Gemini API key in the sidebar for full, context-aware answers."
)


def demo_chat(question: str) -> str:
    q = question.lower()
    for keys, answer in DEMO_CHAT_RULES:
        if any(k in q for k in keys):
            return answer
    return DEMO_CHAT_DEFAULT


# --------------------------------------------------------------------------- #
# Planning & valuation model (indicative, Tier-1 base rates, INR)
# --------------------------------------------------------------------------- #
STRUCTURE_TYPES = {
    #                                     cost/sqft, market/sqft, steel kg/sqft, cement bags/sqft
    "RCC Residential Apartment":        (2200, 9500, 4.0, 0.45),
    "Premium Villa / Bungalow":         (3200, 14000, 4.5, 0.50),
    "Commercial Office (RCC Frame)":    (2800, 12000, 4.5, 0.50),
    "Industrial Shed (PEB Steel)":      (1600, 3500, 7.0, 0.30),
    "Load-Bearing Masonry House":       (1500, 6500, 1.5, 0.55),
}
TIER_COST_MULT = {"Tier 1 (Metro: Mumbai, Delhi, Bengaluru)": 1.00,
                  "Tier 2 (Pune, Jaipur, Nagpur, Raipur)": 0.85,
                  "Tier 3 (Smaller cities & towns)": 0.72}
TIER_MARKET_MULT = {"Tier 1 (Metro: Mumbai, Delhi, Bengaluru)": 1.00,
                    "Tier 2 (Pune, Jaipur, Nagpur, Raipur)": 0.55,
                    "Tier 3 (Smaller cities & towns)": 0.35}
SOFT_COST_PCT = 0.10  # design, approvals, contingency
LAND_SHARE = 0.30     # share of market value attributed to land
MILESTONES = [(15, "Foundation & Excavation"), (30, "Plinth & Ground Beams"),
              (60, "Superstructure (Columns, Slabs)"), (75, "Masonry & Roofing"),
              (90, "MEP, Plaster & Waterproofing"), (100, "Finishing & Handover")]


def compute_project(structure: str, sqft: float, tier: str, pct: float) -> dict:
    cost_rate, mkt_rate, steel_kg, cement_bags = STRUCTURE_TYPES[structure]
    cost_rate *= TIER_COST_MULT[tier]
    mkt_rate *= TIER_MARKET_MULT[tier]
    f = pct / 100.0
    budget = sqft * cost_rate * (1 + SOFT_COST_PCT)
    market_full = sqft * mkt_rate
    asset_now = market_full * (LAND_SHARE + (1 - LAND_SHARE) * f)
    return {
        "cost_rate": cost_rate, "mkt_rate": mkt_rate,
        "budget": budget, "capex": budget * f, "remaining": budget * (1 - f),
        "market_full": market_full, "asset_now": asset_now,
        "steel_total_t": sqft * steel_kg / 1000, "steel_used_t": sqft * steel_kg / 1000 * f,
        "cement_total": sqft * cement_bags, "cement_used": sqft * cement_bags * f,
    }


def build_render_prompt(structure: str, sqft: int, tier: str, pct: int) -> str:
    stage = ("empty plot with site clearing" if pct < 10 else
             "foundation and plinth work in progress" if pct < 30 else
             "reinforced concrete frame under construction with scaffolding" if pct < 60 else
             "structure complete with brickwork and roofing" if pct < 90 else
             "nearly complete with finishing works" if pct < 100 else
             "fully completed, landscaped, golden hour lighting")
    city = "metropolitan Indian city" if "Tier 1" in tier else "mid-size Indian city" if "Tier 2" in tier else "small Indian town"
    return (f"Architectural 3D render of a {structure.lower()}, approx {sqft} sq ft, in a {city}, "
            f"{stage}, photorealistic, wide angle, detailed materials, 8k")


# --------------------------------------------------------------------------- #
# Weather advisory data
# --------------------------------------------------------------------------- #
WEATHER_ADVISORY = {
    "Normal": {
        "icon": "🌤️", "level": "low",
        "headline": "Normal conditions: standard protocols apply",
        "alerts": [
            ("low", "Routine toolbox talk", "Hold 10-minute pre-shift safety briefing and PPE check."),
            ("low", "Housekeeping", "Maintain clear access routes; stack materials within limits."),
            ("low", "Concrete curing", "Standard wet curing for ≥7 days (OPC) per IS 456 cl. 13.5."),
        ],
    },
    "High Wind": {
        "icon": "🌬️", "level": "high",
        "headline": "High wind: suspend crane and work-at-height operations",
        "alerts": [
            ("high", "Suspend crane operations", "Stop lifts when sustained wind ≥ 36 km/h (or per manufacturer). Lower hook, release slew brake, park boom downwind."),
            ("high", "Halt work at height", "No work on scaffolds, roof edges or tower cranes; secure all tools."),
            ("med", "Secure loose materials", "Tie down sheeting, formwork panels, bricks, and bagged cement; remove debris from edges."),
            ("med", "Scaffold & hoarding check", "Verify ties and brace hoardings and temporary structures before resuming."),
        ],
    },
    "Heavy Rain": {
        "icon": "🌧️", "level": "high",
        "headline": "Heavy rain: electrical, excavation and concrete risks",
        "alerts": [
            ("high", "Electrical hazard mitigation", "De-energise temporary supply in wet zones, use RCCB/ELCB (30 mA), raise cable joints off the ground, no wet-hand tool use."),
            ("high", "Excavation & trench safety", "Evacuate trenches, inspect shoring and dewater before re-entry; keep spoil ≥ 1 m from edge."),
            ("med", "Concrete protection", "Do not pour in heavy rain. Cover fresh concrete with tarpaulin; avoid surface washing; re-verify slump and w/c."),
            ("med", "Slip & access control", "Anti-slip walkways, drain site, restrict vehicle movement on soft ground."),
            ("low", "Material storage", "Keep cement on raised pallets under cover; protect steel from corrosion."),
        ],
    },
    "Extreme Heat": {
        "icon": "🔥", "level": "med",
        "headline": "Extreme heat: protect workers and fresh concrete",
        "alerts": [
            ("high", "Heat-stress management", "Mandatory 10-15 min breaks each hour, shaded rest area, ORS/water every 20 min, stop heavy work 12:00-15:00."),
            ("med", "Concrete curing protocol", "Pour in early morning/night, keep concrete <32 °C at placement, use chilled water/retarders, cure ≥10 days with continuous wet hessian/ponding (IS 456 cl. 13.5)."),
            ("med", "Plastic shrinkage control", "Windbreaks, evaporation retardant film, start curing immediately after finishing."),
            ("low", "Steel & formwork", "Shade bars/shuttering; wet formwork before pouring to avoid water loss."),
        ],
    },
}

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "chat" not in st.session_state:
    st.session_state.chat = [{
        "role": "assistant",
        "content": "Hello! I'm **SiteMentor**, your site engineering assistant. Ask about IS 456, curing, "
                   "formwork, scheduling, material quantities or site safety.",
    }]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🏗️ Control Panel")
    api_key_input = st.text_input(
        "Google Gemini API Key", type="password", placeholder="AIza…",
        help="Stored only in this browser session; never written to disk. "
             "On Streamlit Cloud you may set GEMINI_API_KEY in Secrets instead.",
    )
    api_key = (api_key_input or get_secret_key()).strip()
    model_name = st.text_input("Model", value=DEFAULT_MODEL, help="Any multimodal Gemini model id.")
    LIVE = bool(api_key)
    if LIVE:
        st.success("Live AI mode enabled", icon="✅")
    else:
        st.warning("Demonstration mode: simulated outputs", icon="🧪")
    st.caption("Get a free key at aistudio.google.com/apikey")

    st.divider()
    st.markdown("### 🧭 Features")
    st.markdown(
        "1. **Site Safety & PPE Audit**\n2. **Crack Diagnostics**\n"
        "3. **Planning & Valuation**\n4. **Daily Site Assistant**"
    )

    with st.expander("🧱 Architecture overview", expanded=False):
        st.markdown(
            """
```
 Streamlit UI (tabs)
      │
 Service layer ── Gemini (google-genai)
      │             • multimodal JSON analysis
      │             • multi-turn chat
      ├─ Demo engine (offline fallback)
      ├─ Valuation model (pure Python)
      └─ Pollinations AI (image URL)
 State: st.session_state only
```
Zero database. Stateless & 1-click deployable.
"""
        )
    st.caption("Skill Wallet · Capstone Project 1")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """<div class="hero"><h1>🏗️ Multimodal Construction Intelligence Platform</h1>
    <p>AI-powered safety audits, structural diagnostics, project valuation and weather-aware site advisory.</p></div>""",
    unsafe_allow_html=True,
)
if not LIVE:
    st.markdown(
        '<div class="demo-banner">🧪 <b>Demonstration mode:</b> results are simulated. '
        "Enter a Gemini API key in the sidebar for real image analysis.</div>",
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "🦺 Site Safety & PPE", "🧱 Crack Diagnostics", "📐 Planning & Valuation", "🌦️ Daily Site Assistant",
])

# --------------------------------------------------------------------------- #
# TAB 1: PPE audit
# --------------------------------------------------------------------------- #
def normalise_ppe(d: dict) -> dict:
    score = int(clamp(d.get("safety_score"), 0, 100, 0))
    d["safety_score"] = score
    d.setdefault("overall_status", "Compliant" if score >= 85 else "Needs Improvement" if score >= 65 else "Non-Compliant")
    d["ppe_detection"] = d.get("ppe_detection") or []
    hz = []
    for h in d.get("hazards") or []:
        sev = str(h.get("severity", "Low")).title()
        h["severity"] = sev if sev in SEVERITY_RANK else "Low"
        hz.append(h)
    d["hazards"] = sorted(hz, key=lambda h: -SEVERITY_RANK[h["severity"]])
    return d


def render_ppe(d: dict) -> None:
    score = d["safety_score"]
    status = d.get("overall_status", "N/A")
    color = "green" if status == "Compliant" else "amber" if status == "Needs Improvement" else "red"
    hazards = d["hazards"]
    crit = sum(1 for h in hazards if h["severity"] in ("Critical", "High"))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Safety Score", f"{score}/100", "OSHA / IS 7969 weighted"), unsafe_allow_html=True)
    c2.markdown(card("Overall Status", pill(status, color), "Compliance verdict"), unsafe_allow_html=True)
    c3.markdown(card("Workers Detected", str(d.get("workers_detected", "–")), "in frame"), unsafe_allow_html=True)
    c4.markdown(card("High / Critical Hazards", str(crit), f"{len(hazards)} hazards total"), unsafe_allow_html=True)
    st.progress(score / 100)
    if d.get("summary"):
        st.info(d["summary"])

    st.markdown("#### 🦺 PPE Detection Checklist")
    for item in d["ppe_detection"]:
        s = item.get("status", "Not Visible")
        col = "green" if s == "Detected" else "red" if s == "Missing" else "grey"
        a, b, c = st.columns([2, 1.2, 4])
        a.markdown(f"**{item.get('item', '?')}**")
        b.markdown(pill(s, col), unsafe_allow_html=True)
        pct = clamp(item.get("compliance_pct"), 0, 100, 0)
        c.progress(pct / 100, text=f"{int(pct)}% · {item.get('note', '')}")

    st.markdown("#### ⚠️ Identified Site Hazards")
    if not hazards:
        st.success("No hazards identified.")
    for h in hazards:
        sev = h["severity"]
        css = "high" if sev in ("High", "Critical") else "med" if sev == "Medium" else "low"
        st.markdown(
            f'<div class="alert {css}">{pill(sev, SEVERITY_PILL[sev])} &nbsp;<b>{h.get("hazard", "")}</b>'
            f'<br><span class="small-note">Standard: {h.get("standard", "–")}</span>'
            f'<br>🔧 <b>Remedy:</b> {h.get("remedy", "–")}</div>',
            unsafe_allow_html=True,
        )


with tab1:
    st.subheader("Site Safety & PPE Compliance Audit")
    st.caption("Upload a construction site photo. Gemini evaluates PPE and hazards against OSHA 1926 and IS standards.")
    up = st.file_uploader("Site photo (JPG / PNG)", type=["jpg", "jpeg", "png"], key="ppe_up")
    if up:
        try:
            img = load_image(up)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            img = None
        if img:
            left, right = st.columns([1, 1.4])
            left.image(img, caption=up.name, use_container_width=True)
            with right:
                if st.button("🔍 Run Safety Audit", type="primary", key="ppe_btn"):
                    with st.spinner("Analysing site photo…"):
                        try:
                            raw = gemini_json(api_key, model_name, PPE_PROMPT, img) if LIVE else demo_ppe(seed_from(up))
                            st.session_state.ppe_result = normalise_ppe(raw)
                            st.session_state.ppe_name = up.name
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                res = st.session_state.get("ppe_result")
                if res and st.session_state.get("ppe_name") == up.name:
                    render_ppe(res)
                    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                              "file": up.name, "mode": "live" if LIVE else "demo", **res}
                    st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2),
                                       file_name="ppe_safety_report.json", mime="application/json")
    else:
        st.info("Upload a photo to begin.")

# --------------------------------------------------------------------------- #
# TAB 2: Crack diagnostics
# --------------------------------------------------------------------------- #
def normalise_crack(d: dict) -> dict:
    d["estimated_crack_width_mm"] = round(clamp(d.get("estimated_crack_width_mm"), 0, 100, 0.0), 2)
    d["confidence_pct"] = int(clamp(d.get("confidence_pct"), 0, 100, 0))
    r = str(d.get("structural_risk", "Low")).title()
    d["structural_risk"] = r if r in SEVERITY_RANK else "Low"
    d["root_causes"] = d.get("root_causes") or []
    d["recommended_repairs"] = d.get("recommended_repairs") or []
    return d


def render_crack(d: dict) -> None:
    risk = d["structural_risk"]
    urg = d.get("urgency", "–")
    urg_col = "crit" if urg == "Immediate" else "red" if "2 Weeks" in urg else "amber" if "3 Months" in urg else "green"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Defect Type", d.get("defect_type", "–"), f"{d['confidence_pct']}% confidence"), unsafe_allow_html=True)
    c2.markdown(card("Crack Width", f"{d['estimated_crack_width_mm']} mm", f"Pattern: {d.get('crack_pattern', '–')}"), unsafe_allow_html=True)
    c3.markdown(card("Structural Risk", pill(risk, SEVERITY_PILL[risk]), "Engineering classification"), unsafe_allow_html=True)
    c4.markdown(card("Urgency", pill(urg, urg_col), "Recommended action window"), unsafe_allow_html=True)

    w = d["estimated_crack_width_mm"]
    if w > 0.3:
        st.warning(f"Estimated width {w} mm exceeds the ~0.3 mm general serviceability guide of IS 456 cl. 35.3.2.")

    a, b = st.columns(2)
    with a:
        st.markdown("#### 🔎 Probable Root Causes")
        for cause in d["root_causes"]:
            st.markdown(f"- {cause}")
        st.markdown("#### 📈 Monitoring Plan")
        st.write(d.get("monitoring_plan", "–"))
    with b:
        st.markdown("#### 🛠️ Recommended Engineering Repairs")
        for r in d["recommended_repairs"]:
            if isinstance(r, dict):
                st.markdown(f"**{r.get('method', 'Repair')}**  \n{r.get('detail', '')}")
            else:
                st.markdown(f"- {r}")
    if d.get("engineer_note"):
        st.caption(f"📝 {d['engineer_note']}")


with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    st.caption("Upload a concrete / masonry surface photo to classify defects and get repair guidance.")
    up2 = st.file_uploader("Concrete or wall image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="crack_up")
    if up2:
        try:
            img2 = load_image(up2)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            img2 = None
        if img2:
            left, right = st.columns([1, 1.6])
            left.image(img2, caption=up2.name, use_container_width=True)
            with right:
                if st.button("🧪 Diagnose Defect", type="primary", key="crack_btn"):
                    with st.spinner("Running structural diagnostics…"):
                        try:
                            raw = gemini_json(api_key, model_name, CRACK_PROMPT, img2) if LIVE else demo_crack(seed_from(up2))
                            st.session_state.crack_result = normalise_crack(raw)
                            st.session_state.crack_name = up2.name
                        except Exception as e:
                            st.error(f"Diagnosis failed: {e}")
                res = st.session_state.get("crack_result")
                if res and st.session_state.get("crack_name") == up2.name:
                    render_crack(res)
                    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                              "file": up2.name, "mode": "live" if LIVE else "demo", **res}
                    st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2),
                                       file_name="crack_diagnostic_report.json", mime="application/json")
        st.caption("⚠️ AI output is advisory only; confirm with a licensed structural engineer before repairs.")
    else:
        st.info("Upload a surface image to begin.")

# --------------------------------------------------------------------------- #
# TAB 3: Planning & valuation
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Project Planning & Real Estate Valuation")
    c1, c2 = st.columns(2)
    structure = c1.selectbox("Structure type", list(STRUCTURE_TYPES))
    tier = c2.selectbox("Location tier", list(TIER_COST_MULT))
    c3, c4 = st.columns(2)
    sqft = c3.number_input("Built-up area (sq ft)", min_value=100, max_value=1_000_000, value=2000, step=100)
    pct = c4.slider("Completion (%)", 0, 100, 40)

    try:
        p = compute_project(structure, float(sqft), tier, float(pct))
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(card("Total Project Budget", format_inr(p["budget"]), f"₹{p['cost_rate']:,.0f}/sq ft + {int(SOFT_COST_PCT*100)}% soft costs"), unsafe_allow_html=True)
        m2.markdown(card("Capex Incurred", format_inr(p["capex"]), f"Remaining {format_inr(p['remaining'])}"), unsafe_allow_html=True)
        m3.markdown(card("Current Asset Valuation", format_inr(p["asset_now"]), f"Completed value {format_inr(p['market_full'])}"), unsafe_allow_html=True)
        gain = p["asset_now"] - p["capex"]
        m4.markdown(card("Value vs Spend", format_inr(abs(gain)), "Equity created" if gain >= 0 else "Capex ahead of value"), unsafe_allow_html=True)

        st.markdown("#### 🧱 Material Requirements")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Steel (total)", f"{p['steel_total_t']:,.1f} t", f"{p['steel_used_t']:,.1f} t consumed", delta_color="off")
        s2.metric("Steel remaining", f"{p['steel_total_t'] - p['steel_used_t']:,.1f} t")
        s3.metric("Cement (total)", f"{p['cement_total']:,.0f} bags", f"{p['cement_used']:,.0f} used", delta_color="off")
        s4.metric("Cement remaining", f"{p['cement_total'] - p['cement_used']:,.0f} bags")

        st.markdown("#### 🏁 Milestone Progress")
        st.progress(pct / 100, text=f"{pct}% complete")
        cols = st.columns(len(MILESTONES))
        prev = 0
        for col, (limit, name) in zip(cols, MILESTONES):
            done = pct >= limit
            active = prev <= pct < limit
            icon = "✅" if done else "🔨" if active else "⏳"
            col.markdown(f"{icon} **{limit}%**<br><span class='small-note'>{name}</span>", unsafe_allow_html=True)
            prev = limit
        st.caption(
            "Indicative model: construction cost and market rates are Tier-1 baselines scaled by tier. "
            f"Asset value = {int(LAND_SHARE*100)}% land share + {int((1-LAND_SHARE)*100)}% built value × completion. "
            "Not a certified valuation."
        )
    except Exception as e:
        st.error(f"Calculation error: {e}")

    st.markdown("#### 🎨 Conceptual 3D Architectural Rendering")
    default_prompt = build_render_prompt(structure, int(sqft), tier, pct)
    render_prompt = st.text_area("Render prompt (editable)", value=default_prompt, height=80)
    if st.button("Generate Concept Render", key="render_btn"):
        st.session_state.render_url = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(render_prompt)
            + f"?width=1024&height=576&nologo=true&seed={abs(hash(render_prompt)) % 100000}"
        )
    if st.session_state.get("render_url"):
        st.image(st.session_state.render_url, caption="Generated via Pollinations AI (conceptual only)", use_container_width=True)
        st.caption("If the image does not load, the free Pollinations service may be busy; click generate again.")

# --------------------------------------------------------------------------- #
# TAB 4: Weather & assistant
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    weather = st.radio("Today's site weather", list(WEATHER_ADVISORY), horizontal=True)
    adv = WEATHER_ADVISORY[weather]
    st.markdown(f"### {adv['icon']} {adv['headline']}")
    for level, title, text in adv["alerts"]:
        st.markdown(
            f'<div class="alert {level}">{pill({"high":"HIGH","med":"MEDIUM","low":"ADVISORY"}[level], {"high":"red","med":"amber","low":"green"}[level])}'
            f" &nbsp;<b>{title}</b><br>{text}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 💬 SiteMentor: Engineering Assistant")
    head_l, head_r = st.columns([5, 1])
    head_l.caption(f"Context-aware of today's weather: **{weather}**")
    if head_r.button("Clear chat"):
        st.session_state.chat = st.session_state.chat[:1]
        st.rerun()

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about curing, formwork, IS codes, safety, quantities…"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    history = [m for m in st.session_state.chat if m is not st.session_state.chat[0]]
                    reply = gemini_chat(api_key, model_name, history, weather) if LIVE else demo_chat(prompt)
                except Exception as e:
                    reply = f"⚠️ The assistant hit an error: `{e}`. Check your API key / model name and try again."
            st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

st.markdown("---")
st.caption("AI outputs are advisory. Always verify with qualified engineers and applicable codes before acting.")
