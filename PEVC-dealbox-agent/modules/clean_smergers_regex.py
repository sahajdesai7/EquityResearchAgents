import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSON_DIR = PROJECT_ROOT / "json"
INPUT_FILE = JSON_DIR / "smergers_data.json"
OUTPUT_FILE = JSON_DIR / "smergers_data_regex_cleaned.json"

# Exchange Rates (approximate)
EXCHANGE_RATES = {
    "EUR": 1.0, "USD": 0.92, "GBP": 1.17, "INR": 0.011,
    "SGD": 0.68, "THB": 0.026, "NOK": 0.088, "SEK": 0.089
}

# Unit Multipliers to convert to Millions
UNIT_MULTIPLIERS = {
    "thousand": 0.001, "k": 0.001,
    "million": 1.0, "mn": 1.0, "m": 1.0,
    "billion": 1000.0, "bn": 1000.0,
    "crore": 10.0, "cr": 10.0,
    "lakh": 0.1, "lac": 0.1
}

# --- REGEX PATTERNS ---

# 1. Money: Captures Currency + Value + Unit
RE_MONEY = re.compile(r"([A-Za-z]{3})\s+([\d\.,]+)\s+([A-Za-z]+)", re.IGNORECASE)

# 2. EBITDA: Captures ranges "30 - 40" or singles "20"
RE_EBITDA_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
RE_EBITDA_SINGLE = re.compile(r"(\d+(?:\.\d+)?)")

# 3. Percentage: Captures "15%" or "15 %"
RE_PERCENTAGE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# --- HELPER FUNCTIONS ---

def parse_location(loc_str: str) -> Dict[str, Optional[str]]:
    """Splits 'City, Country' or 'City, State, Country'"""
    if not loc_str:
        return {"city": None, "state": None, "country": None}
    
    parts = [p.strip() for p in loc_str.split(',')]
    
    if len(parts) == 2:
        return {"city": parts[0], "state": None, "country": parts[1]}
    elif len(parts) >= 3:
        return {"city": parts[0], "state": parts[1], "country": parts[-1]}
    else:
        return {"city": loc_str, "state": None, "country": None}

def parse_ebitda(ebitda_str: str) -> Dict[str, Optional[float]]:
    """Extracts min, max, avg from strings like '10 - 20 %'"""
    if not ebitda_str:
        return {"min": None, "max": None, "avg": None}
    
    range_match = RE_EBITDA_RANGE.search(ebitda_str)
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        return {"min": min_val, "max": max_val, "avg": round((min_val + max_val) / 2, 2)}
    
    single_match = RE_EBITDA_SINGLE.search(ebitda_str)
    if single_match:
        val = float(single_match.group(1))
        return {"min": val, "max": val, "avg": val}

    return {"min": None, "max": None, "avg": None}

def parse_and_convert_money(money_str: str) -> Dict[str, Any]:
    """Extracts raw money parts and converts to Million EUR"""
    result = {
        "currency": None, "value_raw": None, "unit_raw": None,
        "value_eur_millions": None
    }
    
    if not money_str:
        return result

    match = RE_MONEY.search(money_str)
    if match:
        currency = match.group(1).upper()
        value_raw = float(match.group(2).replace(',', ''))
        unit_raw = match.group(3).lower()
        
        result["currency"] = currency
        result["value_raw"] = value_raw
        result["unit_raw"] = unit_raw

        unit_factor = 0.000001 
        for key, factor in UNIT_MULTIPLIERS.items():
            if key in unit_raw:
                unit_factor = factor
                break
        
        ex_rate = EXCHANGE_RATES.get(currency, 1.0)
        result["value_eur_millions"] = round(value_raw * unit_factor * ex_rate, 3)

    return result

def parse_stake_percentage(item: dict) -> float:
    """
    Determines stake % using a Waterfall Logic:
    1. Explicit field 'stake_sale_percentage'
    2. Hidden in 'purchase_consideration' (e.g., 'for 15%')
    3. Infer from 'stake_sale_category' (Business for Sale = 100%)
    """
    raw_perc = item.get("stake_sale_percentage", "")
    consideration = item.get("purchase_consideration", "")
    category = item.get("stake_sale_category", "").lower()

    # Priority 1: Check the explicit percentage field
    match = RE_PERCENTAGE.search(str(raw_perc))
    if match:
        return float(match.group(1))

    # Priority 2: Check inside purchase consideration string (e.g. "... for 15%")
    match_hidden = RE_PERCENTAGE.search(str(consideration))
    if match_hidden:
        return float(match_hidden.group(1))

    # Priority 3: Infer from Category
    if "business for sale" in category or "full sale" in category or "asset sale" in category:
        return 100.0
    
    return None

# --- MAIN EXECUTION ---

def clean_data():
    if INPUT_FILE.exists():
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        print("Input file not found. Please ensure smergers_data.json exists.")
        return

    cleaned_listings = []
    
    for item in data.get("listings", []):
        # 1. Location
        loc = parse_location(item.get("location"))
        
        # 2. EBITDA
        ebitda = parse_ebitda(item.get("ebitda_margin"))
        
        # 3. Financials
        sales = parse_and_convert_money(item.get("run_rate_sales"))
        price = parse_and_convert_money(item.get("purchase_consideration"))
        
        # 4. Stake Percentage (New Logic)
        stake_pct = parse_stake_percentage(item)

        cleaned_obj = {
            "business_name": item.get("business_name"),
            "city": loc["city"],
            "state": loc["state"],
            "country": loc["country"],
            
            "ebitda_margin_min": ebitda["min"],
            "ebitda_margin_max": ebitda["max"],
            "ebitda_margin_avg": ebitda["avg"],
            
            "sales_currency": sales["currency"],
            "sales_value_raw": sales["value_raw"],
            "sales_unit_raw": sales["unit_raw"],
            "sales_converted_eur_millions": sales["value_eur_millions"],
            
            "price_currency": price["currency"],
            "price_value_raw": price["value_raw"],
            "price_unit_raw": price["unit_raw"],
            "price_converted_eur_millions": price["value_eur_millions"],
            
            "stake_sale_percentage": stake_pct,
            "contact_url": item.get("contact_url")
        }
        cleaned_listings.append(cleaned_obj)

    # Save Output
    OUTPUT_DIR = OUTPUT_FILE.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"listings": cleaned_listings}, f, indent=4)
        
    print(f"Successfully cleaned {len(cleaned_listings)} records.")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Print preview
    if cleaned_listings:
        print("\n--- Preview of First Record ---")
        print(json.dumps(cleaned_listings[0], indent=2))

if __name__ == "__main__":
    clean_data()