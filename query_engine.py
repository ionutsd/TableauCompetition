import re
import os
import json
import requests
import pandas as pd
from data_loader import load_data, COUNTRY_ALIASES, NUMERIC_COLS, get_column_context

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_years(text: str) -> list[int]:
    """Pull all 4-digit years from text."""
    raw = re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", text)
    return [int(y) for y in raw]


def extract_country(text: str, countries: list[str]) -> str | None:
    """Return the first matching country name or alias found in text."""
    lower = text.lower()
    # Try direct match first (longest first to avoid 'iran' matching 'ukraine')
    for c in sorted(countries, key=len, reverse=True):
        if c.lower() in lower:
            return c
    # Try aliases
    for alias, canonical in COUNTRY_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def detect_metric(text: str) -> str:
    """Return the most relevant numeric column name from the query."""
    lower = text.lower()
    for keyword, col in NUMERIC_COLS.items():
        if keyword in lower:
            return col
    return "gasoline_usd_per_liter"  # safe default


def apply_filters(df: pd.DataFrame, years: list[int], country: str | None) -> pd.DataFrame:
    subset = df.copy()
    if years:
        subset = subset[subset["year"].isin(years)]
    if country:
        subset = subset[subset["country"] == country]
    return subset


def fmt(value, decimals=3) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Pandas handlers  (return plain English strings)
# ---------------------------------------------------------------------------

def handle_highest(subset: pd.DataFrame, metric: str, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    row = subset.loc[subset[metric].idxmax()]
    return (
        f"The highest {metric.replace('_', ' ')} {context_desc} was "
        f"**{fmt(row[metric])}** in **{row['country']}** in **{int(row['year'])}**."
    )


def handle_lowest(subset: pd.DataFrame, metric: str, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    row = subset.loc[subset[metric].idxmin()]
    return (
        f"The lowest {metric.replace('_', ' ')} {context_desc} was "
        f"**{fmt(row[metric])}** in **{row['country']}** in **{int(row['year'])}**."
    )


def handle_average(subset: pd.DataFrame, metric: str, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    avg = subset[metric].mean()
    return (
        f"The average {metric.replace('_', ' ')} {context_desc} was "
        f"**{fmt(avg)}**."
    )


def handle_compare(subset: pd.DataFrame, metric: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    grouped = (
        subset.groupby("country")[metric]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    lines = [f"Average {metric.replace('_', ' ')} by country:"]
    for _, row in grouped.iterrows():
        lines.append(f"  - {row['country']}: {fmt(row[metric])}")
    return "\n".join(lines)


def handle_subsidy(subset: pd.DataFrame, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    subsidized = subset[subset["subsidy_flag"] == 1]["country"].unique().tolist()
    heavy = subset[subset["subsidy_regime"] == "heavy"]["country"].unique().tolist()
    if not subsidized:
        return f"No subsidized countries found {context_desc}."
    return (
        f"Subsidized countries {context_desc}: {', '.join(sorted(subsidized))}.\n"
        f"Heavy subsidizers: {', '.join(sorted(heavy)) if heavy else 'none'}."
    )


def handle_oil_producers(subset: pd.DataFrame, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    producers = subset[subset["is_oil_producer"] == 1]["country"].unique().tolist()
    if not producers:
        return f"No major oil producers found {context_desc}."
    return f"Major oil producers {context_desc}: {', '.join(sorted(producers))}."


def handle_trend(subset: pd.DataFrame, metric: str, country: str | None) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    if country:
        data = subset.sort_values("year")[["year", metric]].dropna()
        if data.empty:
            return f"No trend data available for {country}."
        first = data.iloc[0]
        last = data.iloc[-1]
        change = last[metric] - first[metric]
        direction = "increased" if change > 0 else "decreased"
        return (
            f"{country}: {metric.replace('_', ' ')} {direction} from "
            f"**{fmt(first[metric])}** ({int(first['year'])}) to "
            f"**{fmt(last[metric])}** ({int(last['year'])})."
        )
    else:
        grouped = subset.groupby("year")[metric].mean().reset_index().sort_values("year")
        if grouped.empty:
            return "No trend data available."
        first = grouped.iloc[0]
        last = grouped.iloc[-1]
        change = last[metric] - first[metric]
        direction = "increased" if change > 0 else "decreased"
        return (
            f"Global average {metric.replace('_', ' ')} {direction} from "
            f"**{fmt(first[metric])}** ({int(first['year'])}) to "
            f"**{fmt(last[metric])}** ({int(last['year'])})."
        )


def handle_rank(subset: pd.DataFrame, metric: str, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    grouped = (
        subset.groupby("country")[metric]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    lines = [f"Ranking by {metric.replace('_', ' ')} {context_desc}:"]
    for i, row in grouped.iterrows():
        lines.append(f"  {i+1}. {row['country']}: {fmt(row[metric])}")
    return "\n".join(lines)


def handle_price_tier(subset: pd.DataFrame, context_desc: str) -> str:
    if subset.empty:
        return "No data found for that filter combination."
    counts = subset.groupby("price_tier")["country"].nunique().reset_index()
    counts.columns = ["price_tier", "countries"]
    order = ["Very Low (<$0.30)", "Low ($0.30–$0.70)", "Medium ($0.70–$1.10)",
             "High ($1.10–$1.60)", "Very High (>$1.60)"]
    counts["sort"] = counts["price_tier"].apply(
        lambda x: order.index(x) if x in order else 99
    )
    counts = counts.sort_values("sort")
    lines = [f"Price tier distribution {context_desc}:"]
    for _, row in counts.iterrows():
        lines.append(f"  - {row['price_tier']}: {row['countries']} country/ies")
    return "\n".join(lines)


def handle_specific_value(subset: pd.DataFrame, metric: str, country: str, years: list[int]) -> str:
    if subset.empty:
        return f"No data found for {country} in the requested year(s)."
    rows = subset[["year", "country", metric]].dropna().sort_values("year")
    if rows.empty:
        return f"No {metric.replace('_', ' ')} data available for {country}."
    lines = [f"{metric.replace('_', ' ')} for {country}:"]
    for _, row in rows.iterrows():
        lines.append(f"  {int(row['year'])}: {fmt(row[metric])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini fallback
# ---------------------------------------------------------------------------

def call_groq(question: str, subset: pd.DataFrame) -> str:
    if not GROQ_API_KEY:
        return "AI assistant not configured."

    sample_size = min(60, len(subset))
    sample = subset.sample(sample_size, random_state=42) if len(subset) > sample_size else subset
    data_summary = sample[
        ["year", "country", "region", "gasoline_usd_per_liter",
         "diesel_usd_per_liter", "gasoline_real_2024usd",
         "subsidy_regime", "price_tier", "is_oil_producer",
         "tax_pct_of_pump_price"]
    ].to_csv(index=False)

    prompt = f"""You are a fuel price data analyst assistant.
{get_column_context()}
Here is a sample of the relevant data (CSV format):
{data_summary}
Answer the following question concisely and factually, based only on the data provided above.
If you cannot answer from the data, say so clearly.
Do not make up numbers.
Question: {question}
"""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "AI assistant timed out. Please try again."
    except Exception as e:
        return "AI assistant temporarily unavailable. Please try again later."


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def answer_question(question: str, filter_years: list[int] = None) -> dict:
    """
    Main entry point.
    filter_years: years passed from the Tableau filter (optional).
    Returns {"answer": str, "engine": "pandas" | "gemini", "filters": dict}
    """
    df = load_data()
    q = question.lower().strip()

    # 1. Extract years — from query text OR from Tableau filter
    query_years = extract_years(q)
    years = query_years or (filter_years or [])

    # 2. Extract country
    country = extract_country(q, df["country"].tolist())

    # 3. Detect metric
    metric = detect_metric(q)

    # 4. Apply filters
    subset = apply_filters(df, years, country)

    # Build a readable context description for answer strings
    year_desc = f"in {years}" if years else "across all years"
    country_desc = f"for {country}" if country else "across all countries"
    context_desc = f"{year_desc} {country_desc}".strip()

    filters_used = {"years": years, "country": country, "metric": metric}

    # 5. Keyword routing — pandas handles everything it can
    if any(w in q for w in ["highest", "most expensive", "maximum", "max", "priciest"]):
        return {"answer": handle_highest(subset, metric, context_desc),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["lowest", "cheapest", "minimum", "min"]):
        return {"answer": handle_lowest(subset, metric, context_desc),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["average", "avg", "mean"]):
        return {"answer": handle_average(subset, metric, context_desc),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["rank", "ranking", "top 5", "top 10", "list all", "list countries"]):
        return {"answer": handle_rank(subset, metric, context_desc),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["compare", "comparison", "versus", " vs "]):
        return {"answer": handle_compare(subset, metric),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["subsid", "subsidized", "subsidy"]):
        return {"answer": handle_subsidy(subset, context_desc),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["oil producer", "producer", "producing"]):
        return {"answer": handle_oil_producers(subset, context_desc),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["trend", "over time", "evolution", "historical", "changed"]):
        return {"answer": handle_trend(subset, metric, country),
                "engine": "pandas", "filters": filters_used}

    if any(w in q for w in ["price tier", "tier", "category", "bracket"]):
        return {"answer": handle_price_tier(subset, context_desc),
                "engine": "pandas", "filters": filters_used}

    # Specific country + year lookup (e.g. "what was gasoline in Germany in 1995?")
    if country and years:
        return {"answer": handle_specific_value(subset, metric, country, years),
                "engine": "pandas", "filters": filters_used}

    # 6. Gemini fallback for everything else
    groq_answer = call_groq(question, subset)
return {"answer": groq_answer, "engine": "gemini", "filters": filters_used}
