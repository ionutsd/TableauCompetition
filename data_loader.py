import pandas as pd
from pathlib import Path

_df_cache = None

COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "uae": "UAE",
    "emirates": "UAE",
    "south korea": "South Korea",
    "korea": "South Korea",
    "russia": "Russia",
    "saudi": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "venezuela": "Venezuela",
    "nigeria": "Nigeria",
    "south africa": "South Africa",
    "s africa": "South Africa",
}

NUMERIC_COLS = {
    "gasoline": "gasoline_usd_per_liter",
    "petrol": "gasoline_usd_per_liter",
    "gas": "gasoline_usd_per_liter",
    "gasoline price": "gasoline_usd_per_liter",
    "diesel": "diesel_usd_per_liter",
    "diesel price": "diesel_usd_per_liter",
    "lpg": "lpg_usd_per_liter",
    "real gasoline": "gasoline_real_2024usd",
    "inflation adjusted": "gasoline_real_2024usd",
    "real price": "gasoline_real_2024usd",
    "adjusted": "gasoline_real_2024usd",
    "tax": "tax_pct_of_pump_price",
    "tax percentage": "tax_pct_of_pump_price",
    "pretax": "pretax_price_usd_ltr",
    "crude oil": "crude_oil_usd_per_barrel",
    "crude": "crude_oil_usd_per_barrel",
    "oil price": "crude_oil_usd_per_barrel",
    "refinery margin": "refinery_margin_usd_ltr",
    "margin": "refinery_margin_usd_ltr",
    "yoy": "gasoline_yoy_pct_change",
    "year over year": "gasoline_yoy_pct_change",
    "change": "gasoline_yoy_pct_change",
}


def load_data(csv_path: str = "all_countries_combined.csv") -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    _df_cache = pd.read_csv(path)
    return _df_cache


def get_column_context() -> str:
    return """
Dataset: Global fuel prices for 25 countries, years 1924–2024.
Columns:
- year: int (1924-2024)
- country: one of 25 countries (Argentina, Australia, Brazil, Canada, China, France, Germany, India, Indonesia, Iran, Italy, Japan, Mexico, Nigeria, Norway, Russia, Saudi Arabia, Singapore, South Africa, South Korea, Turkey, UAE, United Kingdom, United States, Venezuela)
- region: Latin America, Asia Pacific, North America, Asia, Europe, Middle East, Africa, Eurasia, Europe/Asia
- decade: e.g. "1990s"
- gasoline_usd_per_liter: nominal gasoline price in USD/liter
- gasoline_usd_per_gallon: nominal gasoline price in USD/gallon
- gasoline_real_2024usd: inflation-adjusted gasoline price (2024 USD)
- diesel_usd_per_liter: nominal diesel price in USD/liter
- diesel_real_2024usd: inflation-adjusted diesel price (2024 USD)
- lpg_usd_per_liter: LPG price in USD/liter
- gasoline_local_currency: price in local currency
- gasoline_yoy_pct_change: year-over-year % change in gasoline price
- diesel_yoy_pct_change: year-over-year % change in diesel price
- price_tier: Very Low (<$0.30), Low ($0.30–$0.70), Medium ($0.70–$1.10), High ($1.10–$1.60), Very High (>$1.60)
- tax_pct_of_pump_price: % of pump price that is tax
- pretax_price_usd_ltr: price before tax in USD/liter
- subsidy_flag: 1 = subsidized, 0 = not
- subsidy_regime: heavy, partial, none
- is_oil_producer: 1 = major oil producer, 0 = not
- crude_oil_usd_per_barrel: global crude oil price that year
- crude_oil_usd_per_liter: crude oil price per liter
- refinery_margin_usd_ltr: refinery margin in USD/liter
- oil_price_shock_idx: index of oil price shock (1.0 = no shock)
- us_cpi: US Consumer Price Index
- inflation_deflator_2024: deflator used to convert to 2024 USD
"""
