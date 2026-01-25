import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSON_DIR = PROJECT_ROOT / "json"
INPUT_FILE = JSON_DIR / "smergers_data_genai_cleaned.json"  # Uses the output from previous step
OUTPUT_FILE = JSON_DIR / "dealbox_candidates.json"

# Load Environment Variables
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")
model_id = os.getenv("GROQ_MODEL_ID") or "llama-3.3-70b-versatile"

if not api_key:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

client = Groq(api_key=api_key)

# --- PYDANTIC SCHEMA FOR GEOGRAPHY CHECK ---
class GeoCheck(BaseModel):
    is_in_asia_or_north_america: bool = Field(
        ..., 
        description="True if the city/country is located in the continent of Asia or North America. False otherwise."
    )

# --- FILTERING LOGIC ---

def check_financial_criteria(listing: dict) -> bool:
    """
    Applies the Deal Box Math Logic.
    
    Logic:
    (Partial Stake > 10% AND EBITDA > 20% AND Sales > 0.15M)
    OR
    (Full Sale (100%) AND Sales > 1.0M)
    """
    try:
        # Extract fields with safe defaults
        stake_pct = listing.get("stake_sale_percentage") or 100.0
        ebitda = listing.get("ebitda_margin_avg") or 0.0
        sales_eur = listing.get("sales_converted_eur_millions") or 0.0
        
        # Condition 1: High Quality Partial Stake
        # Note: "Partial Stake" implies < 100%. We check range 10 < x < 100.
        is_partial = 10.0 < stake_pct < 100.0
        cond_1 = is_partial and (ebitda > 20.0) and (sales_eur > 0.15)
        
        # Condition 2: Big Ticket Full Sale
        # "Business sale or Asset sale" implies roughly 100% stake.
        is_full_sale = stake_pct >= 99.0 # Allow slight float margin
        cond_2 = is_full_sale
        
        return cond_1 or cond_2

    except Exception as e:
        print(f"Skipping {listing.get('business_name')}: Missing Data ({e})")
        return False

def check_geography_with_groq(city: str, country: str) -> bool:
    """
    Asks LLM if the location is in Asia or North America.
    """
    if not country: 
        return False
        
    location_str = f"{city}, {country}" if city else country
    
    prompt = f"""
    Is the location "{location_str}" in the continent of Asia or North America?
    Return JSON: {{ "is_in_asia_or_north_america": true/false }}
    """
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = completion.choices[0].message.content
        result = GeoCheck.model_validate_json(content)
        return result.is_in_asia_or_north_america
        
    except Exception as e:
        print(f"Geo Check Error for {location_str}: {e}")
        return False

# --- MAIN EXECUTION ---
def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading cleaned data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    listings = data.get("listings", [])
    dealbox = []

    print(f"Screening {len(listings)} listings...")

    for item in listings:
        name = item.get("business_name", "Unknown")
        
        # 1. Financial Filter (Fast Python Check)
        if check_financial_criteria(item):
            print(f"  [PASS FINANCIAL] {name}")
            
            # 2. Geography Filter (Slower LLM Check)
            city = item.get("city")
            country = item.get("country")
            
            if check_geography_with_groq(city, country):
                print(f"    -> [PASS GEO] Added to Deal Box.")
                
                # 3. Format Output for Buy Side Analyst
                dealbox_item = {
                    "business_name": name,
                    "city": city,
                    "country": country,
                    "ebitda_margin_avg": item.get("ebitda_margin_avg"),
                    "sales_converted_eur_millions": item.get("sales_converted_eur_millions"),
                    "price_converted_eur_millions": item.get("price_converted_eur_millions"),
                    "stake_sale_percentage": item.get("stake_sale_percentage"),
                    "contact_url": item.get("contact_url")
                }
                dealbox.append(dealbox_item)
            else:
                print(f"    -> [FAIL GEO] Location not in Asia/NA.")
                
            time.sleep(0.2) # Polite rate limit
        else:
            # print(f"  [FAIL FINANCIAL] {name}")
            pass

    # Save Results
    OUTPUT_DIR = OUTPUT_FILE.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"dealbox_candidates": dealbox}, f, indent=4)
        
    print(f"\nSearch Complete. Found {len(dealbox)} Deal Box candidates.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()