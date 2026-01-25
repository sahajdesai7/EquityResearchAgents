import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSON_DIR = PROJECT_ROOT / "json"
INPUT_FILE = JSON_DIR / "smergers_data.json"
OUTPUT_FILE = JSON_DIR / "smergers_data_genai_cleaned.json"

# Load Environment Variables
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")
model_id = os.getenv("GROQ_MODEL_ID")

if not api_key:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

if not model_id:
    print("Warning: GROQ_MODEL_ID not found in .env, defaulting to 'llama-3.3-70b-versatile'")
    model_id = "llama-3.3-70b-versatile"

# --- INITIALIZE CLIENT (GROQ) ---
client = Groq(api_key=api_key)

# --- MATH CONSTANTS ---
EXCHANGE_RATES = {
    "EUR": 1.0, "USD": 0.92, "GBP": 1.17, "INR": 0.011,
    "SGD": 0.68, "THB": 0.026, "NOK": 0.088, "SEK": 0.089
}

UNIT_MULTIPLIERS = {
    "thousand": 0.001, "k": 0.001,
    "million": 1.0, "mn": 1.0, "m": 1.0,
    "billion": 1000.0, "bn": 1000.0,
    "crore": 10.0, "cr": 10.0,
    "lakh": 0.1, "lac": 0.1
}

# --- PYDANTIC SCHEMA ---
class ParsedListing(BaseModel):
    city: Optional[str] = Field(None, description="City name inferred from location")
    state: Optional[str] = Field(None, description="State/Province inferred from location")
    country: Optional[str] = Field(None, description="Country name inferred from location")
    
    ebitda_min: Optional[float] = Field(None, description="Minimum EBITDA %")
    ebitda_max: Optional[float] = Field(None, description="Maximum EBITDA %")
    
    sales_currency: Optional[str] = Field(None, description="Currency code (e.g. EUR, USD)")
    sales_value: Optional[float] = Field(None, description="Numeric value of sales")
    sales_unit: Optional[str] = Field(None, description="Unit string (e.g. 'Million', 'K', 'Crore')")
    
    price_currency: Optional[str] = Field(None, description="Currency code (e.g. EUR, USD)")
    price_value: Optional[float] = Field(None, description="Numeric value of price")
    price_unit: Optional[str] = Field(None, description="Unit string")
    
    stake_percentage: Optional[float] = Field(None, description="Percentage of stake sold. Infer from context if missing (e.g. 'Business for sale' = 100).")

# --- HELPER: MATH ENGINE ---
def calculate_metrics(parsed: ParsedListing) -> dict:
    """Performs deterministic math on the data parsed by the LLM."""
    
    # 1. EBITDA Average
    ebitda_avg = None
    if parsed.ebitda_min is not None and parsed.ebitda_max is not None:
        ebitda_avg = round((parsed.ebitda_min + parsed.ebitda_max) / 2, 2)
    elif parsed.ebitda_min is not None:
        ebitda_avg = parsed.ebitda_min # If single value

    # 2. Conversion Helper
    def convert(val, unit, curr):
        if val is None or curr is None: return None
        
        unit_clean = unit.lower().strip() if unit else ""
        factor = 0.000001 # Default to absolute
        
        for k, v in UNIT_MULTIPLIERS.items():
            if k in unit_clean:
                factor = v
                break
        
        rate = EXCHANGE_RATES.get(curr.upper(), 1.0)
        return round(val * factor * rate, 3)

    return {
        "ebitda_margin_avg": ebitda_avg,
        "sales_converted_eur_millions": convert(parsed.sales_value, parsed.sales_unit, parsed.sales_currency),
        "price_converted_eur_millions": convert(parsed.price_value, parsed.price_unit, parsed.price_currency)
    }

# --- CORE FUNCTION ---
def clean_record_with_groq(record: dict) -> dict:
    # 1. Prepare Schema for the Prompt (Required for robust JSON mode on Groq)
    schema_definition = json.dumps(ParsedListing.model_json_schema(), indent=2)

    system_prompt = f"""
    You are an expert Data Parser. 
    You must output valid JSON strictly matching the following schema:
    {schema_definition}
    """

    user_prompt = f"""
    Extract structured data from this business listing.
    
    Raw Record:
    {json.dumps(record)}

    Guidelines:
    1. Location: Split intelligently into City, State, Country.
    2. EBITDA: Extract numerical min/max. If single value ("20%"), min=20, max=20.
    3. Stake %: If explicit ("15%"), use it. If "Business for Sale" or "Asset Sale", assume 100.0.
    4. Financials: Extract raw currency, value, and unit text exactly as seen.
    """

    try:
        # 2. Call Groq API
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        # 3. Parse JSON Response
        content = completion.choices[0].message.content
        parsed_obj = ParsedListing.model_validate_json(content)
        
        # 4. Run Python Math
        math_results = calculate_metrics(parsed_obj)

        return {
            "business_name": record.get("business_name"),
            "city": parsed_obj.city,
            "state": parsed_obj.state,
            "country": parsed_obj.country,
            "ebitda_margin_min": parsed_obj.ebitda_min,
            "ebitda_margin_max": parsed_obj.ebitda_max,
            "ebitda_margin_avg": math_results["ebitda_margin_avg"],
            "sales_currency": parsed_obj.sales_currency,
            "sales_value_raw": parsed_obj.sales_value,
            "sales_unit_raw": parsed_obj.sales_unit,
            "sales_converted_eur_millions": math_results["sales_converted_eur_millions"],
            "price_currency": parsed_obj.price_currency,
            "price_value_raw": parsed_obj.price_value,
            "price_unit_raw": parsed_obj.price_unit,
            "price_converted_eur_millions": math_results["price_converted_eur_millions"],
            "stake_sale_percentage": parsed_obj.stake_percentage,
            "contact_url": record.get("contact_url")
        }

    except Exception as e:
        print(f"Error processing {record.get('business_name')[:15]}...: {e}")
        return record

# --- MAIN EXECUTION ---
def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_listings = []
    listings = data.get("listings", [])
    
    print(f"Processing {len(listings)} records with Groq ({model_id})...")
    
    for i, item in enumerate(listings, 1):
        print(f"[{i}/{len(listings)}] Parsing: {item.get('business_name')[:40]}...")
        cleaned = clean_record_with_groq(item)
        cleaned_listings.append(cleaned)
        
        # Groq is very fast, but a tiny sleep prevents generic rate limits
        time.sleep(0.2) 

    OUTPUT_DIR = OUTPUT_FILE.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"listings": cleaned_listings}, f, indent=4)
        
    print(f"\nSuccess! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()