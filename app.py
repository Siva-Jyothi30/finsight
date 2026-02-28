"""
app.py - FinSight Personal Finance Analytics Agent
===================================================
The Streamlit entry point that ties together every module:
  data_loader -> categorizer -> visualizations -> agent

Layout
------
  Sidebar      : data controls (sample size, fraud filter)
  Page 1 (📊 Dashboard) : KPI tiles + 4 interactive Plotly charts
  Page 2 (🤖 AI Agent)  : conversational chat with the FinSight agent

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from data_loader    import load_data
from categorizer    import categorize, get_summary, CATEGORY_COLORS, CATEGORY_ORDER
from visualizations import (
    monthly_spending_trend,
    category_breakdown_pie,
    top_merchants_bar,
    spending_heatmap,
    category_stacked_bar,
)
from agent import build_context_summary, build_agent, FinSightAgent


# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSight | Personal Finance Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS for dark-card KPI tiles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* KPI metric cards */
div[data-testid="metric-container"] {
    background-color: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 16px 20px;
}
/* Chat message bubbles */
.user-bubble {
    background: #1e3a5f;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #e0e0e0;
}
.agent-bubble {
    background: #1a2d1a;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading, cached per session
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and cleaning transactions …")
def get_data(sample_size: int | None) -> pd.DataFrame:
    """Load + categorize the dataset; cached across reruns."""
    df = load_data(sample_size=sample_size)
    df = categorize(df)
    return df


@st.cache_resource(show_spinner="Initialising AI agent …")
def get_agent(sample_size: int | None) -> FinSightAgent:
    """Build the FinSight agent; cached so the Groq model isn't reinitialised on every rerun."""
    df      = get_data(sample_size)
    context = build_context_summary(df)
    return build_agent(context)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-analytics.png", width=64)
    st.title("FinSight")
    st.caption("Personal Finance Analytics Agent")
    st.divider()

    st.subheader("Data Controls")

    sample_options = {
        "50 000 rows  (fast)":  50_000,
        "200 000 rows (balanced)": 200_000,
        "Full dataset  (slow)": None,
    }
    selected_label = st.selectbox(
        "Dataset size",
        options=list(sample_options.keys()),
        index=0,
        help="Larger samples are more accurate but slower to load.",
    )
    sample_size = sample_options[selected_label]

    include_fraud = st.toggle(
        "Include fraudulent transactions",
        value=False,
        help="Toggle to include the 0.6% of transactions flagged as fraud.",
    )

    st.divider()
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        options=["📊 Dashboard", "🤖 AI Agent"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data: Kaggle Credit Card Transactions  \nModel: llama-3.3-70b via Groq")


# ---------------------------------------------------------------------------
# Load data (applying fraud filter)
# ---------------------------------------------------------------------------

df_full = get_data(sample_size)
df      = df_full if include_fraud else df_full[df_full["is_fraud"] == 0]

# Date range filter (added to sidebar after data is available)
_periods     = pd.period_range(
    start=df["datetime"].min().to_period("M"),
    end=df["datetime"].max().to_period("M"),
    freq="M",
)
_month_labels = [p.strftime("%b %Y") for p in _periods]

with st.sidebar:
    st.divider()
    st.subheader("Date Filter")
    _date_range = st.select_slider(
        "Select range",
        options=_month_labels,
        value=(_month_labels[0], _month_labels[-1]),
    )

_start = pd.to_datetime(_date_range[0])
_end   = pd.to_datetime(_date_range[1]) + pd.offsets.MonthEnd(0)
df     = df[(df["datetime"] >= _start) & (df["datetime"] <= _end)]


# ---------------------------------------------------------------------------
# Page: 📊 Dashboard
# ---------------------------------------------------------------------------

if page == "📊 Dashboard":

    st.title("📊 FinSight Dashboard")
    st.caption(
        f"Showing **{len(df):,}** transactions · "
        f"{df['datetime'].min().strftime('%b %Y')} – {df['datetime'].max().strftime('%b %Y')}"
    )

    # -- KPI row --
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_spend  = df["amount"].sum()
    total_txns   = len(df)
    avg_txn      = df["amount"].mean()
    top_cat      = get_summary(df).iloc[0]["spending_category"]
    top_cat_pct  = 100 * df[df["spending_category"] == top_cat]["amount"].sum() / total_spend

    col1.metric("Total Spend",        f"${total_spend:,.0f}")
    col2.metric("Transactions",       f"{total_txns:,}")
    col3.metric("Avg Transaction",    f"${avg_txn:.2f}")
    col4.metric("Top Category",       top_cat)
    col5.metric("Top Category Share", f"{top_cat_pct:.1f}%")

    st.divider()

    # -- Full-width stacked bar: monthly composition --
    st.subheader("Monthly Spend Overview")
    st.plotly_chart(category_stacked_bar(df), use_container_width=True)

    st.divider()

    # -- Row 1: Monthly trend + Pie --
    st.subheader("Spending Trends")
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.plotly_chart(
            monthly_spending_trend(df),
            use_container_width=True,
        )

    with chart_col2:
        st.plotly_chart(
            category_breakdown_pie(df),
            use_container_width=True,
        )

    st.divider()

    # -- Row 2: Top merchants + Heatmap --
    st.subheader("Merchant & Timing Analysis")
    chart_col3, chart_col4 = st.columns([3, 2])

    with chart_col3:
        top_n = st.slider("Number of merchants to display", min_value=5, max_value=25, value=15, step=5)
        st.plotly_chart(
            top_merchants_bar(df, top_n=top_n),
            use_container_width=True,
        )

    with chart_col4:
        st.plotly_chart(
            spending_heatmap(df),
            use_container_width=True,
        )

    st.divider()

    # -- Category summary table --
    st.subheader("Category Breakdown Table")
    summary = get_summary(df)
    summary["total_spent"]     = summary["total_spent"].map("${:,.2f}".format)
    summary["avg_transaction"] = summary["avg_transaction"].map("${:,.2f}".format)
    summary["transaction_count"] = summary["transaction_count"].map("{:,}".format)
    summary.columns = ["Category", "Total Spent", "Transactions", "Avg Transaction"]
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: 🤖 AI Agent
# ---------------------------------------------------------------------------

elif page == "🤖 AI Agent":

    st.title("🤖 FinSight AI Agent")
    st.caption(
        "Powered by **llama-3.3-70b** via Groq. "
        "Ask anything about your spending data. The agent has all the numbers."
    )

    # Initialise chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []   # list of {"role": "user"|"assistant", "content": str}

    # Load (or reuse cached) agent
    try:
        agent: FinSightAgent = get_agent(sample_size)
    except ValueError as e:
        st.error(f"Agent setup failed: {e}")
        st.stop()

    # -- Suggested starter questions --
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        starters = [
            "What is my total spending and top spending category?",
            "Which merchants am I spending the most at?",
            "When do I spend the most? (day and time)",
            "How has my spending changed month over month?",
            "What percentage of my budget goes to food?",
        ]
        cols = st.columns(len(starters))
        for col, q in zip(cols, starters):
            if col.button(q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking …"):
                    reply = agent.ask(q)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

    # -- Render chat history --
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -- Chat input --
    if user_input := st.chat_input("Ask FinSight about your finances …"):
        # Immediately show the user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                reply = agent.ask(user_input)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    # -- Clear conversation button --
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.messages.clear()
            agent.clear_history()
            st.rerun()
