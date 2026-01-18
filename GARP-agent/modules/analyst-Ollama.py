import os
import time
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.ollama import Ollama 
from ddgs import DDGS 
from tqdm import tqdm 

# Load environment variables
load_dotenv()

# --- Configuration: Local Inference ---
OLLAMA_MODEL_ID = "llama3.2:3b"

# Define Output Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# --- Helper Functions ---

def get_company_name(ticker):
    """Safely fetches the full company name from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info 
        return info.get('longName', ticker)
    except Exception as e:
        tqdm.write(f"⚠️ Could not fetch company name for {ticker}: {e}")
        return ticker

def search_duckduckgo(query: str, max_results=1) -> str:
    """Executes the search via Python and returns a pre-formatted string."""
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No search results found for: {query}"
            
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
    Returns an Agent configured with Local DeepSeek via Ollama.
    """
    return Agent(
        model=Ollama(id=OLLAMA_MODEL_ID), 
        description="You are a cynical, forensic financial analyst.",
        instructions=[
            "You are investigating a stock for a potential investment.",
            "Your goal is to find the TRUTH, not just repeat the marketing hype.",
            "Analyze the provided Search Results to answer the user's request.",
            "Always cite your sources using the links provided in the results.",
        ],
        debug_mode = True,
        markdown=True,
    )

def safe_run(agent, initial_search_query, analysis_instruction, sleep_time=1):
    """
    Helper: Manages the Agentic "Think-Act-Observe-React" Cycle.
    Max Loops: 2 (Initial Search + 1 Optional Follow-up)
    """
    current_query = initial_search_query
    accumulated_context = ""
    
    # We allow a maximum of 2 cycles (Step 1 -> Step 6 -> Step 7 -> Stop)
    max_cycles = 2
    
    for cycle in range(max_cycles):
        # --- STEP 1 / 6: ACTION (Search) ---
        tqdm.write(f"   🔎 Cycle {cycle+1}: Searching '{current_query}'...")
        
        # Python performs the search (Tools are manual)
        new_results = search_duckduckgo(current_query)
        
        # Append to memory so the agent sees BOTH searches if a second one happens
        accumulated_context += f"\n\n=== SEARCH RESULTS (Round {cycle+1}) ===\n{new_results}\n==============================\n"

        # --- STEP 2: CONTEXT PREPARATION ---
        # If it's the first cycle, we give the agent the option to REACT (Search again)
        if cycle == 0:
            prompt = f"""
            {analysis_instruction}

            {accumulated_context}

            Step 1: Analyze the search results above.
            Step 2: Decide if you have enough information to answer the instruction.
            
            [DECISION LOGIC]
            - IF specific critical information is missing: Reply ONLY with the word "SEARCH:" followed by a better query. 
              Example: SEARCH: NVDA lawsuits 2024 details
            - IF the information is sufficient: Provide the final analysis and report immediately.
            """
        else:
            # Final Cycle: Force an answer using whatever we have
            prompt = f"""
            {analysis_instruction}

            {accumulated_context}

            You have gathered all available information. Please provide the final analysis now.
            """

        # --- STEP 3/4/5: THINK & OBSERVE ---
        try:
            time.sleep(sleep_time)
            response = agent.run(prompt)
            
            if not response or not response.content:
                tqdm.write("⚠️ Empty response from model. Retrying...")
                continue

            content = response.content.strip()

            # --- STEP 6: REACT (Check if Agent wants to search again) ---
            # We look for the "SEARCH:" keyword at the start of the response
            if cycle == 0 and content.upper().startswith("SEARCH:"):
                # Extract the new query
                new_query = content.split(":", 1)[1].strip()
                tqdm.write(f"   🧠 Agent requests new search: '{new_query}'")
                
                # Update the query and loop to Cycle 2
                current_query = new_query
                continue 
            
            # If no search requested (or we are in the final cycle), return the analysis
            return response

        except Exception as e:
            tqdm.write(f"❌ Error during cycle {cycle+1}: {e}")
            return None
            
    return None

def analyze_sentiment_and_news(ticker):
    """
    Step 3, 4 & 5: Full 5-Step Deep Dive (Local LLM Version).
    """
    print(f"\n🕵️‍♂️ Analyst Agent ({OLLAMA_MODEL_ID}): Starting 5-Step Deep Dive for {ticker}...")
    
    company_name = get_company_name(ticker)
    print(f"   🏢 Identified Company: {company_name}")

    # ✅ Start Timer
    start_time = time.time()

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

    # ✅ Progress Bar
    with tqdm(total=5, desc="Initializing...", unit="step") as pbar:

        # --- LINK 1: Historical Context ---
        pbar.set_description("Step 1/5: Historical Context")
        resp_1 = safe_run(
            agent, 
            initial_search_query=f"{company_name} {ticker} corporate strategy events {current_year-5}..{current_year-2}",
            analysis_instruction=f"Summarize major corporate events or strategy shifts for '{company_name}' between {current_year-5} and {current_year-2}."
        )
        if resp_1: research_notes["HISTORY"] = resp_1.content
        pbar.update(1)

        # --- LINK 2: Red Flags ---
        pbar.set_description("Step 2/5: Red Flags Check")
        resp_2 = safe_run(
            agent,
            initial_search_query=f"{company_name} lawsuit fraud SEC investigation short seller report",
            analysis_instruction=f"List any lawsuits, fraud allegations, or SEC investigations for '{company_name}'. If clean, state 'No major red flags found'."
        )
        if resp_2: research_notes["RED_FLAGS"] = resp_2.content
        pbar.update(1)

        # --- LINK 3: Recent Drivers ---
        pbar.set_description("Step 3/5: Recent Drivers")
        resp_3 = safe_run(
            agent,
            initial_search_query=f"{company_name} earnings growth drivers stock news {current_year-2}..{current_year}",
            analysis_instruction=f"Summarize the latest earnings results, growth drivers, and reasons for stock movement for '{company_name}' from {current_year-2} to present."
        )
        if resp_3: research_notes["RECENT"] = resp_3.content
        pbar.update(1)

        # --- LINK 4: Competitive Moat ---
        pbar.set_description("Step 4/5: Moat Analysis")
        resp_4 = safe_run(
            agent,
            initial_search_query=f"{company_name} competitive advantage moat network effect switching costs",
            analysis_instruction=f"""
            Analyze the competitive advantages of '{company_name}'. Look for:
            1. Intangible Assets (Brands, Patents)
            2. Switching Costs
            3. Network Effects
            4. Cost Advantages
            """
        )
        if resp_4: research_notes["MOAT"] = resp_4.content
        pbar.update(1)

        # --- LINK 5: Management Quality ---
        pbar.set_description("Step 5/5: Management Audit")
        resp_5 = safe_run(
            agent,
            initial_search_query=f"{company_name} management capital allocation insider ownership governance",
            analysis_instruction=f"""
            Investigate the management of '{company_name}'. Look for:
            1. Capital Allocation track record
            2. Insider Ownership (Skin in the game)
            3. Governance issues
            """
        )
        if resp_5: research_notes["MANAGEMENT"] = resp_5.content
        pbar.update(1)
        
        pbar.set_description("✅ Analysis Complete")

    # ✅ Stop Timer & Calculate Duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(timedelta(seconds=int(elapsed_time)))
    print(f"\n⏱️ Total Execution Time: {formatted_time}")

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
            f.write(f"Model: {OLLAMA_MODEL_ID} (Local)\n")
            f.write(f"Execution Time: {formatted_time}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            sections = ["HISTORY", "RED_FLAGS", "RECENT", "MOAT", "MANAGEMENT"]
            for section in sections:
                f.write(f"[PART {sections.index(section)+1}: {section.replace('_', ' ')}]\n")
                content = research_notes.get(section)
                if content is None: content = "Data unavailable."
                f.write(str(content) + "\n\n")
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