import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.groq import Groq
from ddgs import DDGS 

# Load environment variables
load_dotenv()

# --- Configuration: Dynamic Key Management ---
# Default to a strong model like Llama 3 70b
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

# 1. Load all available keys into a list
API_KEYS = []

# Check primary key
if os.getenv("GROQ_API_KEY"):
    API_KEYS.append(os.getenv("GROQ_API_KEY"))

# Check secondary keys (2 through 6)
for i in range(2, 7):
    key = os.getenv(f"GROQ_API_KEY{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    raise ValueError("❌ No GROQ_API_KEYs found in .env file. Please add at least one.")

print(f"🔑 Loaded {len(API_KEYS)} Groq API Keys.")

# 2. Track the current key index
CURRENT_KEY_INDEX = 0

# Define Output Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# --- Helper Functions ---

def get_current_key():
    """Returns the currently active API key."""
    return API_KEYS[CURRENT_KEY_INDEX]

def rotate_api_key():
    """
    Updates the global index to the next key in the list.
    Returns the new key.
    """
    global CURRENT_KEY_INDEX
    # Modulo operator % ensures we loop back to 0 after the last key
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    print(f"\n🔄 Switching to API Key #{CURRENT_KEY_INDEX + 1}...")
    return API_KEYS[CURRENT_KEY_INDEX]

def get_company_name(ticker):
    """Safely fetches the full company name from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info 
        return info.get('longName', ticker)
    except Exception as e:
        print(f"⚠️ Could not fetch company name for {ticker}: {e}")
        return ticker

def search_duckduckgo_news(query: str) -> str:
    """Searches DuckDuckGo for the specified query."""
    try:
        # Utilizing the imported DDGS wrapper from your previous code
        results = DDGS().text(query, max_results=1)
        if not results:
            return f"No results found for: {query}"
            
        output = []
        for article in results:
            title = article.get('title', 'N/A')
            body = article.get('body', 'N/A')
            url = article.get('href', 'N/A')
            output.append(f"Title: {title}\nSummary: {body}\nLink: {url}\n---")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"

def get_market_analyst():
    """
    Returns an Agent configured with Groq using the CURRENT key.
    """
    return Agent(
        model=Groq(id=GROQ_MODEL_ID, api_key=get_current_key()),
        tools=[search_duckduckgo_news],
        description="You are a cynical, forensic financial analyst.",
        instructions=[
            "You are investigating a stock for a potential investment.",
            "Your goal is to find the TRUTH, not just repeat the marketing hype.",
            "Use the `search_duckduckgo_news` tool to find factual articles.",
            "DO NOT attempt to open files, read URLs, or use tools named 'open_file'.",
            "Relay the information provided in the search results.",
            "Always cite your sources with links provided by the tool.",
        ],
        markdown=True,
    )

def safe_run(agent, prompt, sleep_time=2):
    """
    Helper: Runs the agent with retry logic.
    TRIGGERS KEY ROTATION on 429/Exhausted errors and retries the SAME query.
    """
    # Construct the single query
    query = f"stock market news and analysis {prompt}"

    # We try the query up to 3 times (in case of network blips or need to swap key)
    attempt = 0
    while attempt < 3:
        try:
            time.sleep(sleep_time)
            response = agent.run(query)

            # Check if response has content and no error
            if response and response.content:
                # Groq responses are sometimes raw strings, sometimes structured.
                # Unlike Gemini, we don't always need to parse JSON for errors manually here,
                # but we catch exceptions below.
                return response
            
            # If no content, break the loop
            break

        except Exception as e:
            error_msg = str(e).lower()
            print(f"Exception message: {error_msg}")

            # Check for Quota/Rate Limit errors (Groq typically throws 429)
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                # Extract retry delay from error message if available (default to 5s)
                retry_delay = 5.0
                retry_delay_match = re.search(r'try again in (\d+\.\d+)s', error_msg)
                if retry_delay_match:
                    retry_delay = float(retry_delay_match.group(1))
                
                print(f"⚠️ Rate Limit Hit on Key #{CURRENT_KEY_INDEX + 1}.")
                print(f"🕒 Waiting for {retry_delay} seconds before retrying...")

                # Wait for the suggested retry delay
                time.sleep(retry_delay)

                # 1. ROTATE THE KEY
                new_key = rotate_api_key()

                # 2. HOT SWAP: Update the Agent's Model with the new key
                agent.model = Groq(id=GROQ_MODEL_ID, api_key=new_key)

                print("⚡ Key Swapped. Retrying previous query immediately...")
                continue
            else:
                print(f"❌ Error: {e}")
                attempt += 1  # Increment attempt for non-critical errors

    return None

def analyze_sentiment_and_news(ticker):
    """
    Step 3, 4 & 5: Full 5-Step Deep Dive.
    """
    print(f"\n🕵️‍♂️ Analyst Agent (Groq): Starting 5-Step Deep Dive for {ticker}...")
    
    company_name = get_company_name(ticker)
    print(f"   🏢 Identified Company: {company_name}")

    agent = get_market_analyst()
    current_year = datetime.now().year
    
    # Initialize Memory
    research_notes = {
        "HISTORY": "No data found.",
        "RED_FLAGS": "No data found.",
        "RECENT": "No data found.",
        "MOAT": "No data found.",
        "MANAGEMENT": "No data found."
    }

    # --- LINK 1: Historical Context ---
    print(f"   ⛓️ Link 1: Analyzing Historical Context ({current_year-5}-{current_year-2})...")
    resp_1 = safe_run(agent, f"Find major corporate events or strategy shifts for '{company_name} ({ticker})' between {current_year-5} and {current_year-2}.", 1)
    if resp_1: research_notes["HISTORY"] = resp_1.content

    # --- LINK 2: Red Flags ---
    print(f"   ⛓️ Link 2: Sweeping for Red Flags...")
    resp_2 = safe_run(agent, f"Search for 'lawsuit', 'fraud', 'SEC investigation', 'accounting irregularity', or 'short seller report' regarding '{company_name} ({ticker})'.", 2)
    if resp_2: research_notes["RED_FLAGS"] = resp_2.content

    # --- LINK 3: Recent Drivers ---
    print(f"   ⛓️ Link 3: Analyzing Recent Drivers...")
    resp_3 = safe_run(agent, f"Find the latest earnings results, growth drivers, and stock movement reasons for '{company_name} ({ticker})' from {current_year-2} to present.", 2)
    if resp_3: research_notes["RECENT"] = resp_3.content

    # --- LINK 4: Competitive Moat ---
    print(f"   ⛓️ Link 4: Investigating Competitive Moat...")
    moat_prompt = f"""
    Analyze the competitive advantages of '{company_name} ({ticker})'.
    Search specifically for evidence of:
    1. "Intangible Assets" (Brands, Patents, Regulatory Licenses)
    2. "Switching Costs" (High retention, sticky ecosystem)
    3. "Network Effects" (Value grows with users)
    4. "Cost Advantage" (Scale, proprietary process)
    """
    resp_4 = safe_run(agent, moat_prompt, 2)
    if resp_4: research_notes["MOAT"] = resp_4.content

    # --- LINK 5: Management Quality ---
    print(f"   ⛓️ Link 5: Auditing Management & Governance...")
    mgmt_prompt = f"""
    Investigate the management of '{company_name} ({ticker})'.
    Search for:
    1. "Capital Allocation" track record (buybacks, dividends, M&A history)
    2. "Insider Ownership" (Do they own stock?)
    3. "Related Party Transactions" or governance issues.
    """
    resp_5 = safe_run(agent, mgmt_prompt, 2)
    if resp_5: research_notes["MANAGEMENT"] = resp_5.content

    # --- SAVE TO FILE ---
    print(f"   💾 Saving Raw Research to File...")
    
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    safe_ticker = ticker.replace(".", "_")
    filename = f"research_{safe_ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"ANALYST RESEARCH REPORT FOR {company_name} ({ticker})\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            # Write all sections
            sections = ["HISTORY", "RED_FLAGS", "RECENT", "MOAT", "MANAGEMENT"]
            for section in sections:
                f.write(f"[PART {sections.index(section)+1}: {section.replace('_', ' ')}]\n")
                f.write(research_notes[section] + "\n\n")
                f.write("-" * 30 + "\n\n")
                
            f.write("="*60 + "\n")
            
        print(f"✅ Research saved successfully: {filepath}")
        return filepath

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None

# --- Testing Block ---
if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. NVDA or RELIANCE.NS): ").strip()
    saved_file = analyze_sentiment_and_news(ticker)
    
    if saved_file:
        print(f"\n📄 Research available at: {saved_file}")