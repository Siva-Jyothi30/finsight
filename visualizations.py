"""
visualizations.py - FinSight Personal Finance Analytics Agent
=============================================================
Builds all interactive Plotly charts used by the Streamlit app.
Every function accepts a clean, categorized DataFrame and returns
a ``plotly.graph_objects.Figure``. No Streamlit imports here so
the charts can be used or tested independently.

Charts
------
1. monthly_spending_trend()  : Line chart, total spend per month
2. category_breakdown_pie()  : Pie/donut chart, spend share by category
3. top_merchants_bar()       : Horizontal bar, top N merchants by spend
4. spending_heatmap()        : Heatmap, hour-of-day x day-of-week spend
5. category_stacked_bar()    : Stacked bar, monthly spend composition by category
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from categorizer import CATEGORY_COLORS, CATEGORY_ORDER


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_FONT_FAMILY = "Inter, Helvetica Neue, Arial, sans-serif"
_BG_COLOR    = "#0f1117"   # matches Streamlit dark theme background
_PAPER_COLOR = "#1a1d27"   # slightly lighter card background
_TEXT_COLOR  = "#e0e0e0"

_BASE_LAYOUT = dict(
    font=dict(family=_FONT_FAMILY, color=_TEXT_COLOR, size=13),
    paper_bgcolor=_PAPER_COLOR,
    plot_bgcolor=_BG_COLOR,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)


# ---------------------------------------------------------------------------
# 1. Monthly spending trend (line chart)
# ---------------------------------------------------------------------------

def monthly_spending_trend(df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing total spending per calendar month, broken down
    by spending category so the user can spot seasonal patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Categorized DataFrame (must have ``spending_category``,
        ``amount``, ``year``, ``month``, ``month_name`` columns).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Aggregate: sum of spend per (year, month, category)
    monthly = (
        df.groupby(["year", "month", "month_name", "spending_category"], observed=True)
        .agg(total=("amount", "sum"))
        .reset_index()
        .sort_values(["year", "month"])
    )

    fig = px.line(
        monthly,
        x="month_name",
        y="total",
        color="spending_category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"spending_category": CATEGORY_ORDER},
        markers=True,
        labels={
            "month_name":        "Month",
            "total":             "Total Spent ($)",
            "spending_category": "Category",
        },
        title="Monthly Spending Trends by Category",
    )

    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(
            tickangle=-35,
            gridcolor="#2a2d3a",
            showline=False,
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            gridcolor="#2a2d3a",
            showline=False,
        ),
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Category breakdown (donut pie chart)
# ---------------------------------------------------------------------------

def category_breakdown_pie(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart showing each spending category's share of total spend.

    Parameters
    ----------
    df : pd.DataFrame
        Categorized DataFrame (must have ``spending_category`` and
        ``amount`` columns).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    summary = (
        df.groupby("spending_category", observed=True)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total"})
    )

    fig = px.pie(
        summary,
        names="spending_category",
        values="total",
        color="spending_category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"spending_category": CATEGORY_ORDER},
        hole=0.45,           # donut style
        title="Spending by Category",
    )

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
    )

    # Apply base layout first, then override legend separately to avoid
    # a "multiple values for keyword argument" conflict with _BASE_LAYOUT.
    fig.update_layout(**_BASE_LAYOUT, showlegend=True)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Top merchants (horizontal bar chart)
# ---------------------------------------------------------------------------

def top_merchants_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of the top N merchants ranked by total spend.

    Parameters
    ----------
    df : pd.DataFrame
        Categorized DataFrame (must have ``merchant``, ``amount``,
        ``spending_category`` columns).
    top_n : int
        Number of merchants to display (default 15).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    top = (
        df.groupby(["merchant", "spending_category"], observed=True)
        .agg(total=("amount", "sum"), txn_count=("amount", "count"))
        .reset_index()
        .sort_values("total", ascending=False)
        .head(top_n)
        .sort_values("total", ascending=True)   # ascending so highest bar is at top
    )

    fig = px.bar(
        top,
        x="total",
        y="merchant",
        color="spending_category",
        color_discrete_map=CATEGORY_COLORS,
        orientation="h",
        text="total",
        custom_data=["txn_count", "spending_category"],
        title=f"Top {top_n} Merchants by Total Spend",
        labels={
            "total":             "Total Spent ($)",
            "merchant":          "Merchant",
            "spending_category": "Category",
        },
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Category: %{customdata[1]}<br>"
            "Total Spent: $%{x:,.2f}<br>"
            "Transactions: %{customdata[0]:,}<extra></extra>"
        ),
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#2a2d3a"),
        yaxis=dict(showgrid=False),
        showlegend=True,
        bargap=0.25,
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Spending heatmap: hour of day x day of week
# ---------------------------------------------------------------------------

def spending_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of average transaction amount by hour of day (y-axis)
    and day of week (x-axis). Reveals when spending peaks occur.

    Parameters
    ----------
    df : pd.DataFrame
        Categorized DataFrame (must have ``hour``, ``day_of_week``,
        ``amount`` columns).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Ordered days so Sunday doesn't end up in the middle
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    pivot = (
        df.groupby(["hour", "day_of_week"], observed=True)["amount"]
        .mean()
        .reset_index()
        .pivot(index="hour", columns="day_of_week", values="amount")
        .reindex(columns=day_order)   # enforce day order
        .fillna(0)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Avg $ / txn", tickprefix="$"),
            hovertemplate="Day: %{x}<br>Hour: %{y}:00<br>Avg: $%{z:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Average Spend: Hour of Day × Day of Week",
        xaxis=dict(title="Day of Week", showgrid=False),
        yaxis=dict(
            title="Hour of Day (24h)",
            tickmode="array",
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
            showgrid=False,
            autorange="reversed",
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# 5. Category stacked bar: monthly spend composition
# ---------------------------------------------------------------------------

def category_stacked_bar(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart showing total monthly spend broken down by category.
    Complements the line chart by making the overall budget composition
    visible at a glance alongside month-to-month totals.

    Parameters
    ----------
    df : pd.DataFrame
        Categorized DataFrame (must have ``spending_category``,
        ``amount``, ``year``, ``month``, ``month_name`` columns).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    monthly = (
        df.groupby(["year", "month", "month_name", "spending_category"], observed=True)
        .agg(total=("amount", "sum"))
        .reset_index()
        .sort_values(["year", "month"])
    )

    # Period label "Jan 2019", "Feb 2019" preserves chronological order
    monthly["period"] = monthly["month_name"].str[:3] + " " + monthly["year"].astype(str)

    fig = px.bar(
        monthly,
        x="period",
        y="total",
        color="spending_category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"spending_category": CATEGORY_ORDER},
        barmode="stack",
        labels={
            "period":            "Month",
            "total":             "Total Spent ($)",
            "spending_category": "Category",
        },
        title="Monthly Spend Composition by Category",
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(tickangle=-35, gridcolor="#2a2d3a", showline=False),
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#2a2d3a", showline=False),
        hovermode="x unified",
        bargap=0.2,
    )

    return fig


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_data
    from categorizer import categorize

    df = load_data(sample_size=50_000)
    df = categorize(df)

    print("Building charts ...")

    fig1 = monthly_spending_trend(df)
    fig2 = category_breakdown_pie(df)
    fig3 = top_merchants_bar(df)
    fig4 = spending_heatmap(df)
    fig5 = category_stacked_bar(df)

    # Open each chart in the default browser for visual inspection
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()

    print("All 5 charts rendered successfully.")
