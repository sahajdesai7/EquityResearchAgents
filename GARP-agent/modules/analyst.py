import os
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Agno Imports ---
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.vectordb.lancedb import LanceDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
# We use the raw search library for manual control, not the Agent tool
from ddgs import DDGS

# Load environment variables
load_dotenv()

# --- Configuration ---
OLLAMA_MODEL_ID = "llama3.2:3b"
OLLAMA_EMBEDDER_MODEL = "nomic-embed-text"
MAX_WORKERS = 3  # Adjust based on your CPU (3 is safe for local Ollama)
SEARCH_LIMIT = 5 # Number of results for both Web and KB

# --- 📂 PATH CONFIGURATION 📂 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUTS_DIR = os.path.join(PROJECT_ROOT, "static_inputs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LANCEDB_DIR = os.path.join(INPUTS_DIR, "tmp", "lancedb")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LANCEDB_DIR, exist_ok=True)

# --- 1. HELPER: DOWNLOAD PDF ---
def get_company_name(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('longName', ticker)
    except:
        return ticker

def try_pdf_download(url, pdf_path, timeout=15):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", "").lower():
            with open(pdf_path, "wb") as f: f.write(resp.content)
            return True
        return False
    except: return False

def download_annual_report(ticker, company_name):
    pdf_path = os.path.join(INPUTS_DIR, f"{ticker}_AR.pdf")
    if os.path.exists(pdf_path): 
        return pdf_path

    print(f"📉 Downloading Annual Report for {ticker}...")
    query = f"{company_name} annual report {datetime.now().year - 1} pdf"

    try:
        results = DDGS().text(query, max_results=5)
        for res in results:
            url = res.get("href")
            if url and try_pdf_download(url, pdf_path): 
                print(f"✅ Downloaded: {url}")
                return pdf_path
        print("❌ No downloadable PDF found.")
        return None
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return None

# --- 2. KNOWLEDGE BASE SETUP ---
def setup_knowledge_base(ticker, pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    db_uri = os.path.join(LANCEDB_DIR, "lancedb")
    table_name = f"docs_{ticker}"

    embedder = OllamaEmbedder(id=OLLAMA_EMBEDDER_MODEL, dimensions=768)
    vector_db = LanceDb(table_name=table_name, uri=db_uri, embedder=embedder)
    knowledge = Knowledge(vector_db=vector_db)

    if vector_db.exists():
        pass # DB exists, skip ingestion
    else:
        print(f"⚙️ Ingesting PDF (this happens once)...")
        knowledge.insert(path=pdf_path, reader=PDFReader(chunk=True))
        print("✅ Knowledge Base Created.")

    return knowledge

# --- 3. TASK WORKER (OPTIMIZED) ---
def process_single_task(task, kb, company_name, current_year):
    """
    Runs a single analysis task. 
    Manually fetches Context (KB) and Web (DDGS) to speed up the Agent.
    """
    section_key = task['key']
    context_parts = []

    # --- A. MANUAL KNOWLEDGE BASE SEARCH ---
    # We search the KB for *every* task if a query is provided
    if kb and task.get('kb_query'):
        try:
            relevant_docs = kb.search(task['kb_query'], max_results=SEARCH_LIMIT) 
            if relevant_docs:
                kb_text = "\n---\n".join([doc.content for doc in relevant_docs])
                context_parts.append(f"### INTERNAL ANNUAL REPORT CONTEXT:\n{kb_text}")
        except Exception as e:
            print(f"⚠️ KB Search Warning ({section_key}): {e}")

    # --- B. MANUAL WEB SEARCH ---
    # We search the Web for *every* task if a query is provided
    if task.get('web_query'):
        try:
            # Using DDGS directly avoids the Agent "thinking" about whether to use tools
            web_results = DDGS().text(task['web_query'], max_results=SEARCH_LIMIT)
            if web_results:
                web_text = "\n---\n".join([f"Source: {r['title']}\nSummary: {r['body']}" for r in web_results])
                context_parts.append(f"### EXTERNAL WEB SEARCH RESULTS:\n{web_text}")
        except Exception as e:
            print(f"⚠️ Web Search Warning ({section_key}): {e}")

    # --- C. CONSTRUCT PROMPT ---
    # Combine all contexts into one string
    full_context_str = "\n\n".join(context_parts)
    
    full_instruction = (
        f"{full_context_str}\n\n"
        f"TASK: {task['prompt']}\n"
        "INSTRUCTIONS: Use the provided Internal Context and Web Results to answer. "
        "Cite sources where possible (e.g., 'Internal Report' or 'Web News')."
    )

    # --- D. AGENT EXECUTION ---
    # Note: No tools are passed to the agent. It is purely a text processor now.
    agent = Agent(
        model=Ollama(id=OLLAMA_MODEL_ID),
        description="You are a forensic financial analyst.",
        instructions=[f"Focus on '{company_name}'.", "Be concise. Do not repeat marketing or management hype, be a pragmatist, a realist"],
        debug_mode=True, 
        markdown=True,
    )

    try:
        response = agent.run(full_instruction)
        return section_key, response.content
    except Exception as e:
        return section_key, f"Agent Error: {e}"

# --- 4. MAIN WORKFLOW ---
def analyze_sentiment_and_news(ticker):
    print(f"\n🚀 Starting Fast Deep Dive for {ticker}...")
    company_name = get_company_name(ticker)
    pdf_path = download_annual_report(ticker, company_name)
    kb = setup_knowledge_base(ticker, pdf_path)
    current_year = datetime.now().year
    
    # Define Tasks with EXPLICIT queries for both Web and KB
    tasks = [
        {
            "key": "ANNUAL_REPORT_INSIGHTS",
            "kb_query": "Risk Factors Litigation Management Discussion warnings",
            "web_query": f"'{company_name}' risk factors litigation management warnings {current_year-1} {current_year}",
            "prompt": """
    You are analyzing company risk disclosures.

    INTERNAL (KB):
    - Extract explicit risk factors, warnings, and forward-looking cautionary statements from annual reports or MD&A.
    - Quote or paraphrase conservatively. Do NOT infer risks not stated.

    EXTERNAL (Web):
    - Identify risks or warnings discussed by analysts, regulators, or media in the last 12–24 months.

    OUTPUT:
    1. Internal Reported Risks (bullet list)
    2. External Reported Risks (bullet list)
    3. Alignment Check:
    - Risks mentioned in both
    - Risks only mentioned externally
    - Risks only mentioned internally

    RULES:
    - Do not speculate.
    - If no contradiction exists, explicitly state: "No material divergence found."
    """
        },

        {
            "key": "HISTORY",
            "kb_query": "Corporate strategy acquisitions divestitures history",
            "web_query": f"'{company_name}' acquisitions divestitures strategy changes {current_year-5} to {current_year}",
            "prompt": """
    You are constructing a factual strategic timeline.

    INTERNAL (KB):
    - Extract management-stated strategic actions (M&A, divestitures, restructurings, pivots).

    EXTERNAL (Web):
    - Identify externally reported strategic events or commentary for the same period.

    OUTPUT (Chronological Table):
    - Year
    - Event
    - Source (Internal / External / Both)
    - Strategic Intent (as stated, not inferred)

    RULES:
    - Do not assign success or failure unless explicitly stated by sources.
    - Prefer dates and transaction names over general descriptions.
    """
        },

        {
            "key": "RED_FLAGS",
            "kb_query": "Legal proceedings investigations fraud lawsuits",
            "web_query": f"'{company_name}' lawsuits fraud regulatory investigations settlements {current_year}",
            "prompt": """
    You are performing a risk verification exercise.

    INTERNAL (KB):
    - List disclosed legal proceedings, investigations, or contingent liabilities.

    EXTERNAL (Web):
    - Identify lawsuits, regulatory probes, fines, or settlements reported publicly.

    OUTPUT:
    1. Issues disclosed internally
    2. Issues reported externally
    3. Disclosure Gap Analysis:
    - External issues NOT disclosed internally
    - Internal issues with limited external coverage

    RULES:
    - Do NOT assume wrongdoing.
    - If information is incomplete, label it clearly as "ongoing" or "unresolved."
    """
        },

        {
            "key": "RECENT_PERFORMANCE",
            "kb_query": "Management discussion financial performance earnings results",
            "web_query": f"'{company_name}' earnings analysis analyst commentary stock performance {current_year}",
            "prompt": """
    You are comparing narrative tone between management and analysts.

    INTERNAL (KB):
    - Summarize management's discussion of recent financial performance.

    EXTERNAL (Web):
    - Summarize analyst and market commentary for the same period.

    OUTPUT:
    1. Financial Snapshot (revenue, margins, growth direction — no forecasts)
    2. Management Tone (optimistic / neutral / cautious, with evidence)
    3. Analyst Tone (optimistic / neutral / cautious, with evidence)
    4. Tone Divergence Assessment

    RULES:
    - Do not introduce new financial metrics unless explicitly stated.
    - Avoid forward-looking projections.
    """
        },

        {
            "key": "MOAT_ANALYSIS",
            "kb_query": "Competitive advantage market share competition",
            "web_query": f"'{company_name}' competitive advantage moat market share industry analysis",
            "prompt": """
    You are performing a MOAT ANALYSIS strictly following Pat Dorsey's framework
    from 'The Little Book That Builds Wealth'.

    STEP 1: Identify the PRIMARY moat type (if any), choosing ONLY from:
    - Intangible Assets (brand, patents, licenses, regulatory protection)
    - Switching Costs
    - Network Effects
    - Cost Advantages
    - Efficient Scale

    STEP 2: Evidence Gathering
    INTERNAL (KB):
    - Extract management claims related to competitive advantage.
    - Note exact language used (e.g., pricing power, customer retention, scale).

    EXTERNAL (Web):
    - Extract analyst or industry evidence that supports or challenges these claims.

    STEP 3: Moat Validation (Dorsey Test)
    For the identified moat:
    - Is it structural or cyclical?
    - Does it improve ROIC relative to peers?
    - Is it difficult for competitors to replicate?

    STEP 4: Durability Assessment
    - Moat Strength: None / Narrow / Wide
    - Expected Longevity: <5 years / 5–10 years / 10+ years

    OUTPUT STRUCTURE:
    1. Claimed Moat (Internal)
    2. Observed Moat Evidence (External)
    3. Moat Type (Dorsey classification)
    4. Durability & Threats
    5. Final Verdict (Concise, evidence-based)

    RULES:
    - If no moat exists, explicitly state: "No durable moat identified."
    - Do NOT mix multiple moat types unless clearly supported.
    - Avoid buzzwords. Every claim must tie to evidence.
    """
        }
    ]


    research_notes = {}

    print(f"⚡ Running {len(tasks)} analysis tasks in parallel (Workers: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_single_task, task, kb, company_name, current_year): task['key'] 
            for task in tasks
        }
        
        with tqdm(total=len(tasks), unit="task") as pbar:
            for future in as_completed(future_to_task):
                key, result = future.result()
                research_notes[key] = result
                pbar.update(1)

    # Save to File
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    filename = f"research_{ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# REPORT: {company_name} ({ticker})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        for task in tasks:
            key = task['key']
            content = research_notes.get(key, "Analysis Failed")
            f.write(f"## {key}\n{content}\n\n{'-'*30}\n\n")

    print(f"✅ Research saved: {filepath}")

if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. NVDA): ").strip()
    if ticker:
        analyze_sentiment_and_news(ticker)