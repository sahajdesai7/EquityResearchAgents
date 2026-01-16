import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.groq import Groq
from ddgs import DDGS  # Import the DuckDuckGo news search function

# Load environment variables
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file. Please add it.")

# --- Custom DuckDuckGo News Tool ---
def search_duckduckgo_news(query: str) -> str:
    """
    Searches DuckDuckGo for the specified query.
    Useful for finding latest headlines, earnings, and corporate events.

    Args:
        query (str): The search term (e.g., "NVDA lawsuits", "Reliance earnings").

    Returns:
        str: A formatted list of the top 5 news articles with titles, dates, and links.
    """
    try:
        # Search DuckDuckGo for news
        results = DDGS().text(query, max_results=3)

        if not results:
            return f"No DuckDuckGo News results found for query: {query}"

        output = []
        for article in results:
            title = article.get('title', 'N/A')
            date = article.get('date', 'N/A')
            url = article.get('url', 'N/A')
            # Format cleanly for the LLM
            output.append(f"- {title} ({date}) [Link: {url}]")

        return "\n".join(output)

    except Exception as e:
        return f"Error searching DuckDuckGo News: {e}"

def get_market_analyst():
    """
    Returns an Agent configured with Groq and the custom DuckDuckGo News tool.
    """
    return Agent(
        model=Groq(
            id=GROQ_MODEL_ID,
            api_key=GROQ_API_KEY
        ),
        # 🛠️ Pass the Python function directly as a tool
        tools=[search_duckduckgo_news],
        description="You are a cynical, forensic financial analyst.",
        instructions=[
            "You are investigating a stock for a potential investment.",
            "Your goal is to find the TRUTH, not just repeat the marketing hype.",
            "Use the `search_duckduckgo_news` tool to find factual articles.",
            "DO NOT attempt to open files, read URLs, or use tools named 'open_file",
            "Relay the information provided in the preview fields of the search results.",
            "Always cite your sources with links provided by the tool.",
        ],
        markdown=True,
    )

def safe_run(agent, prompt, sleep_time=2):
    """
    Helper: Runs the agent with retry logic.
    Handles Groq Rate Limits and empty search results.
    """
    max_retries = 3
    # Variations to try if the first query fails
    query_variations = [
        prompt,
        f"{prompt} news",
        f"{prompt} analysis",
        f"{prompt} report"
    ]

    for attempt in range(max_retries):
        for query in query_variations:
            try:
                time.sleep(sleep_time)
                # The prompt is passed as the message; the tool is called internally by the LLM
                return agent.run(query)

            except Exception as e:
                error_msg = str(e)

                # 1. Handle Rate Limits
                if "429" in error_msg or "Rate limit" in error_msg:
                    wait_time = (attempt + 1) * 10
                    print(f"⚠️ Groq Rate Limit Hit. Cooling down for {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    # Break inner loop to retry the SAME query after sleep, or move to next attempt
                    break

                # 2. Handle Logic Errors
                else:
                    print(f"❌ Error during agent run: {e}")
                    if attempt == max_retries - 1:
                        print("⚠️ Max retries reached. Skipping this step.")
                        return None
                    break # Try next attempt

    return None

def analyze_sentiment_and_news(ticker):
    """
    Step 3: Agentic Prompt Chain (Groq + DuckDuckGo News).
    """
    print(f"\n🕵️‍♂️ Analyst Agent (Groq + DuckDuckGo News): Starting 5-Year Deep Dive for {ticker}...")

    agent = get_market_analyst()
    current_year = datetime.now().year

    # --- CHAIN LINK 1: Historical Context ---
    print(f"   ⛓️ Link 1: Analyzing Historical Context ({current_year-5} to {current_year-2})...")
    history_prompt = f"""
    Use the search tool to find major corporate events or strategy shifts for '{ticker}'
    between {current_year-5} and {current_year-2}.
    Summarize the 'Old Narrative' in 3 bullet points.
    """
    safe_run(agent, history_prompt, sleep_time=1)

    # --- CHAIN LINK 2: Red Flags ---
    print(f"   ⛓️ Link 2: Sweeping for Red Flags...")
    red_flag_prompt = f"""
    Use the search tool to find "lawsuit", "fraud", "SEC investigation", or "short seller report" regarding '{ticker}' from the last 5 years.
    List any findings. If clean, state "No major red flags found."
    """
    safe_run(agent, red_flag_prompt, sleep_time=2)

    # --- CHAIN LINK 3: Recent Drivers ---
    print(f"   ⛓️ Link 3: Analyzing Recent Drivers ({current_year-2} to Present)...")
    recent_prompt = f"""
    Use the search tool to find the latest news and earnings results for '{ticker}' from {current_year-2} to present.
    Focus on:
    1. Why is the stock moving NOW?
    2. Recent earnings surprises.
    """
    safe_run(agent, recent_prompt, sleep_time=2)

    # --- CHAIN LINK 4: Synthesis ---
    print(f"   ⛓️ Link 4: Synthesizing Final Report...")
    synthesis_prompt = f"""
    Based on all the research you just performed (Historical, Red Flags, and Recent),
    write a final 'Sentiment & News Report' for {ticker}.

    **Weighting Instructions**:
    - Give 70% importance to the Recent News (Link 3).
    - Give 30% importance to the Historical Context (Link 1).
    - The Red Flags section must be prominent.

    **Required Output Format**:
    ## 📰 News & Sentiment Analysis: {ticker}

    ### 🚀 Growth Drivers (The Bull Case)
    * [Detail 1 with Source]
    * [Detail 2 with Source]

    ### 🚩 Red Flags (The Bear Case)
    * [Details of lawsuits/fraud/resignations found in Link 2]
    * (If none, explicitly state "Clean Forensic Check")

    ### 📜 Historical Context (The Foundation)
    * [One sentence summary of the 2021-2023 period]

    ### 🧠 Analyst Verdict
    * **Sentiment Score**: (Bullish / Neutral / Bearish)
    * **Key Takeaway**: (1 sentence summary)
    """
    response = safe_run(agent, synthesis_prompt, sleep_time=1)

    if response:
        return response.content
    else:
        return "⚠️ Analysis Failed: Groq API limits exhausted or connection error."

# --- Testing Block ---
if __name__ == "__main__":
    print(f"🔹 Connected to Groq Model: {GROQ_MODEL_ID}")

    ticker = input("Enter ticker for News Analysis (e.g. NVDA or RELIANCE.NS): ").strip()
    tool_code = """
import os
from ddgs import ddg_news
# Ensure ddgs is installed and working
try:
    import ddgs
    print("ddgs library is installed and ready.")
except ImportError:
    print("Please run: pip install ddgs")
    """
    report = analyze_sentiment_and_news(ticker)

    print("\n" + "="*50)
    print(report)
    print("="*50)