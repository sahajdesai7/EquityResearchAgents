import os
import json
from pathlib import Path
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "smergers_data.json"

# --- EXECUTION ---
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"Warning: .env file not found at {ENV_PATH}")

api_key = os.getenv("FIRECRAWL_API_KEY")
if not api_key:
    raise ValueError("Please set FIRECRAWL_API_KEY in your .env file")

app = FirecrawlApp(api_key=api_key)

# --- SCHEMA DEFINITION ---
class BusinessListing(BaseModel):
    business_name: str = Field(..., description="The name of the business or the generic headline provided (e.g., 'Newly Established IT Company').")
    # NEW FIELD ADDED HERE
    location: Optional[str] = Field(None, description="The place of business, city, or country mentioned (e.g., 'Mumbai', 'Singapore', 'USA').")
    run_rate_sales: Optional[str] = Field(None, description="Run Rate Sales value including currency and unit (e.g., 'INR 2 Cr', 'USD 500K').")
    ebitda_margin: Optional[str] = Field(None, description="EBITDA margin percentage. If a range is given, include the full range.")
    stake_sale_category: Optional[str] = Field(None, description="The category of the stake sale, typically 'Full Sale' or 'Partial Stake'.")
    purchase_consideration: Optional[str] = Field(None, description="The purchase consideration or asking price.")
    stake_sale_percentage: Optional[str] = Field(None, description="The percentage of the stake being sold.")
    contact_url: Optional[str] = Field(None, description="The specific URL to contact the business or view the full profile.")

class ExtractionSchema(BaseModel):
    listings: List[BusinessListing]

# --- CRAWLING ---
target_url = "https://www.smergers.com/businesses-for-sale-and-investment/b/#"
print(f"Extracting data from {target_url}...")

try:
    response = app.extract(
        urls=[target_url],
        # Updated prompt to explicitly ask for location
        prompt="Extract the first 5 business listings visible on the page with their location, financial details, and contact links.",
        schema=ExtractionSchema.model_json_schema()
    )

    if response.success:
        extracted_content = response.data
        
        # Enforce limit of 5
        if 'listings' in extracted_content and isinstance(extracted_content['listings'], list):
            extracted_content['listings'] = extracted_content['listings'][:5]
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(extracted_content, f, indent=4, ensure_ascii=False)
            
        count = len(extracted_content.get('listings', []))
        print(f"Successfully extracted {count} listings.")
        print(f"Data saved to: {OUTPUT_FILE}")
    else:
        print("Extraction failed:", response)

except Exception as e:
    print(f"An error occurred: {e}")