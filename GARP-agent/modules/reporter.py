import os
import glob
import time  # ✅ Ensure time is imported
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
from tqdm import tqdm  # ✅ Import TQDM

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.ollama import Ollama 

# --- Helper Functions ---

def get_latest_file(ticker: str, prefix: str) -> Optional[str]:
    """Finds the most recent file in 'outputs/'."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    search_pattern = os.path.join(outputs_dir, f"{prefix}*{ticker}*")
    files = glob.glob(search_pattern)
    
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def get_file_content(filepath: str) -> str:
    """Safely reads text content."""
    if not filepath or not os.path.exists(filepath):
        return "Data not available."
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def get_historical_peg(ticker_symbol):
    """Calculates historical P/E and PEG ratios for the last 3 years."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        financials = ticker.financials
        if 'Diluted EPS' not in financials.index:
            return pd.DataFrame() 
            
        eps_series = financials.loc['Diluted EPS'].T.sort_index()
        if len(eps_series) < 4:
            return pd.DataFrame() 
            
        eps_history = eps_series.tail(4)
        data = []
        
        for i in range(1, len(eps_history)):
            date_t = eps_history.index[i]
            eps_t = eps_history.iloc[i]
            eps_prev = eps_history.iloc[i-1]
            
            # A. Growth Rate
            if eps_prev == 0:
                growth_rate = 0
            else:
                growth_rate = (eps_t / eps_prev) - 1
            growth_rate_pct = growth_rate * 100
            
            # B. Price (Approximate at Fiscal Year End)
            start_date = date_t
            end_date = date_t + timedelta(days=5)
            hist = ticker.history(start=start_date, end=end_date)
            price_t = hist['Close'].iloc[0] if not hist.empty else 0
                
            # C. Ratios
            pe_ratio = price_t / eps_t if eps_t > 0 else 0
            peg_ratio = pe_ratio / growth_rate_pct if (growth_rate_pct > 0 and pe_ratio > 0) else 0
            
            data.append({
                "Date": date_t.strftime('%Y-%m-%d'),
                "Price": round(price_t, 2),
                "PE Ratio": round(pe_ratio, 2),
                "PEG Ratio": round(peg_ratio, 2)
            })
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"⚠️ Error calculating PEG: {e}")
        return pd.DataFrame()

# --- Agent Definitions ---

def get_reporter_agent(model_id="llama3.2:3b"):
    return Agent(
        model=Ollama(id=model_id),
        description="You are a Pragmatic Portfolio Manager.",
        instructions=[
            "Your goal is to write a balanced Investment Memo.",
            "Focus on 'Growth at a Reasonable Price' (GARP).",
            "CRITICAL: Do not write a separate 'Risks' section. Interweave risks directly into your thesis.",
            "Example: 'While the PE expansion to 30x suggests optimism, this relies entirely on the new product launch; any delay would likely revert the multiple to 20x.'",
            "Assign scores (0-10) for: News (x3), Moat (x2), Management (x2), Earnings Quality (x3).",
        ],
        debug_mode = True,
        markdown=True
    )

def get_reviewer_agent(model_id="llama3.2:3b"):
    return Agent(
        model=Ollama(id=model_id),
        description="You are a Senior Risk Officer (The Skeptic).",
        instructions=[
            "Review the Draft Memo provided by the Reporter.",
            "Check 1: Is the tone too optimistic? (We want Realism, not Hype).",
            "Check 2: Did they ignore the PEG 'Caution' flag?",
            "Check 3: Are the risks interwoven? If they dumped risks at the end, flag it.",
            "Output a critique summarizing what needs to be fixed. Do not rewrite the full memo yet.",
        ],
        debug_mode = True,
        markdown=True
    )

# --- Main Workflow ---

def generate_investment_memo(ticker):
    print(f"\n🏆 Reporter Agent: Assembling Investment Memo for {ticker}...")
    
    # ✅ Start Timer
    start_time = time.time()
    
    # We have 5 distinct steps in our manual progress bar
    with tqdm(total=5, desc="Initializing", unit="step") as pbar:
        
        # --- PHASE 1: Intelligence Gathering ---
        pbar.set_description("Step 1/5: Gathering Intelligence")
        
        analyst_file = get_latest_file(ticker, "research_")
        forensic_file = get_latest_file(ticker, "forensic_")
        chart_file = get_latest_file(ticker, "charts_")
        
        analyst_data = get_file_content(analyst_file)
        forensic_data = get_file_content(forensic_file)
        
        # Calculate PEG
        peg_df = get_historical_peg(ticker)
        peg_context = "PEG Data Unavailable"
        caution_flag = "None"
        
        if not peg_df.empty:
            current_peg = peg_df['PEG Ratio'].iloc[-1]
            hist_avg_peg = peg_df['PEG Ratio'].mean()
            peg_context = peg_df.to_markdown(index=False)
            
            if current_peg < 1.5:
                if current_peg > (hist_avg_peg * 1.2):
                    caution_flag = f"⚠️ CAUTION: PEG ({current_peg}) is attractive (<1.5) but rising steeply vs history ({hist_avg_peg:.2f}). Window closing."
                else:
                    caution_flag = "✅ GREEN: Valuation is attractive and stable."
            else:
                caution_flag = f"❌ RED: PEG ({current_peg}) exceeds GARP limit of 1.5."
        
        pbar.update(1) # Done with Prep

        # --- PHASE 2: The Draft (Reporter) ---
        pbar.set_description("Step 2/5: Drafting Thesis")
        
        reporter = get_reporter_agent()
        draft_prompt = f"""
        Write a Draft Investment Memo for {ticker}.
        
        === INPUTS ===
        [Market Research]:
        {analyst_data[:2000]}... (truncated for brevity)
        
        [Forensic Analysis]:
        {forensic_data}
        
        [Valuation Context (PEG History)]:
        {peg_context}
        
        [Valuation Flag]: {caution_flag}
        ==============
        
        Task:
        1. Score the opportunity (News, Moat, Mgmt, Quality).
        2. Write the thesis. Focus on: Is the PE expansion sustainable?
        3. INTERWEAVE the risks. If you say "Growth is good", immediately add "But X could derail it."
        """
        draft_response = reporter.run(draft_prompt)
        draft_text = draft_response.content
        
        pbar.update(1) # Done with Drafting

        # --- PHASE 3: The Critique (Reviewer) ---
        pbar.set_description("Step 3/5: Risk Officer Review")
        
        reviewer = get_reviewer_agent()
        critique_response = reviewer.run(f"Review this draft:\n\n{draft_text}\n\nDid they respect the Caution Flag: {caution_flag}?")
        critique_text = critique_response.content
        
        pbar.update(1) # Done with Review

        # --- PHASE 4: The Final Polish (Reporter) ---
        pbar.set_description("Step 4/5: Finalizing Memo")
        
        final_prompt = f"""
        Refine the Investment Memo based on this critique:
        "{critique_text}"
        
        Ensure the final output is formatted as clean HTML/Markdown.
        Include a link to the charts: {chart_file}
        """
        final_response = reporter.run(final_prompt)
        
        pbar.update(1) # Done with Final Polish
        
        # --- PHASE 5: Save Output ---
        pbar.set_description("Step 5/5: Saving File")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "outputs", f"Investment_Memo_{ticker}.md")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_response.content)
            
        pbar.update(1) # Done!
        pbar.set_description("✅ Complete")

    # ✅ Stop Timer & Print Duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(timedelta(seconds=int(elapsed_time)))
    
    print(f"\n⏱️ Total Execution Time: {formatted_time}")
    print(f"✅ Memo Saved: {output_path}")
    
    return output_path

if __name__ == "__main__":
    ticker = input("Enter ticker: ").strip()
    generate_investment_memo(ticker)