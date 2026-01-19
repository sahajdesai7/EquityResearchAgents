import os
import time
import shutil
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.ollama import Ollama 
from agno.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.document import Document

from ddgs import DDGS 
from tqdm import tqdm 
from pypdf import PdfReader  # ✅ Using pypdf as requested
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tempfile
import hashlib

# Load environment variables
load_dotenv()

# --- Configuration: Local Inference ---
OLLAMA_MODEL_ID = "llama3.2:3b"
OLLAMA_EMBEDDER_MODEL = "nomic-embed-text" # Ensure you ran: `ollama pull nomic-embed-text`

# Define Output Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
INPUTS_DIR = os.path.join(BASE_DIR, "static_inputs")

# Ensure inputs directory exists
if not os.path.exists(INPUTS_DIR):
    os.makedirs(INPUTS_DIR)

def get_company_name(ticker):
    """Safely fetches the full company name from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info 
        return info.get('longName', ticker)
    except Exception as e:
        return ticker

# --- CUSTOM EMBEDDER (Bypassing Agno Internals) ---

class AgnoOllamaEmbedderAdapter:
    def __init__(self, ollama_embedder, dimensions: int):
        self.ollama = ollama_embedder
        self.dimensions = dimensions

    def get_embedding_and_usage(self, text: str):
        embedding = self.ollama.get_embeddings([text])[0]
        return embedding, {}

class LocalOllamaEmbedder:
    def __init__(self, model: str = OLLAMA_EMBEDDER_MODEL, dimensions: int = 768):
        self.model = model
        self.dimensions = dimensions
        self.api_url = "http://localhost:11434/api/embeddings"

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text via HTTP request."""
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")
            return [0.0] * self.dimensions

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
        return embeddings

def try_pdf_download(url, pdf_path, timeout=15):
    """
    Tries to download a PDF from a URL.
    - Works with direct PDFs
    - Handles redirects
    - Extracts embedded PDFs from HTML pages
    Returns True if PDF saved successfully, else False.
    """

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers=headers,
            allow_redirects=True
        )

        if resp.status_code != 200:
            return False

        content_type = resp.headers.get("Content-Type", "").lower()

        # ✅ Case 1: Direct PDF (even if URL doesn't end with .pdf)
        if "application/pdf" in content_type:
            with open(pdf_path, "wb") as f:
                f.write(resp.content)

            tqdm.write(f"      ✅ PDF downloaded: {url}")
            return True

        # ✅ Case 2: HTML page → extract embedded PDF links
        if "text/html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.select("a[href]"):
                href = a["href"].strip()

                if ".pdf" not in href.lower():
                    continue

                pdf_url = urljoin(url, href)

                try:
                    pdf_resp = requests.get(
                        pdf_url,
                        timeout=timeout,
                        headers=headers,
                        allow_redirects=True
                    )

                    if (
                        pdf_resp.status_code == 200
                        and "application/pdf" in pdf_resp.headers.get("Content-Type", "").lower()
                    ):
                        with open(pdf_path, "wb") as f:
                            f.write(pdf_resp.content)

                        tqdm.write(f"      ✅ Embedded PDF downloaded: {pdf_url}")
                        return True

                except Exception:
                    continue

        return False

    except Exception as e:
        tqdm.write(f"      ⚠️ PDF download error ({url}): {e}")
        return False

def download_annual_report(ticker, company_name):
    """
    Attempts to download the latest Annual Report PDF via DuckDuckGo.
    """
    pdf_path = os.path.join(INPUTS_DIR, f"{ticker}_AR.pdf")

    if os.path.exists(pdf_path):
        return True

    tqdm.write(f"   📉 Attempting to download Annual Report for {ticker}...")

    current_year = datetime.now().year
    query = f"{company_name} annual report {current_year - 1} pdf"

    try:
        results = DDGS().text(query, max_results=10)

        for res in results:
            url = res.get("href")
            if not url:
                continue

            tqdm.write(f"      🔎 Trying: {url}")

            if try_pdf_download(url, pdf_path):
                return True

        tqdm.write("      ❌ No downloadable PDF found in top results.")
        return False

    except Exception as e:
        tqdm.write(f"      ⚠️ Search error: {e}")
        return False

def get_annual_report_kb(ticker):
    """
    Read PDF using pypdf, chunk by page, and load into LanceDB (Agno) correctly.
    """
    pdf_path = os.path.join(INPUTS_DIR, f"{ticker}_AR.pdf")

    if not os.path.exists(pdf_path):
        return None

    tqdm.write(f"   📄 Found Annual Report: {pdf_path}")

    # ------------------ Read PDF ------------------
    try:
        reader = PdfReader(pdf_path)
        texts = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 100:
                texts.append(f"[Page {i+1}] {text.strip()}")

        if not texts:
            tqdm.write("   ⚠️ PDF extraction yielded no usable text. Skipping.")
            return None

    except Exception as e:
        tqdm.write(f"   ❌ Error reading PDF: {e}")
        return None

        # ------------------ Embedder ------------------
    raw_embedder = LocalOllamaEmbedder(
        model=OLLAMA_EMBEDDER_MODEL,
        dimensions=768
    )

    embedder = AgnoOllamaEmbedderAdapter(
        ollama_embedder=raw_embedder,
        dimensions=768
    )


        # ------------------ LanceDB ------------------
    db_path = os.path.join(tempfile.gettempdir(), "lancedb", ticker)
    os.makedirs(db_path, exist_ok=True)

    vector_db = LanceDb(
        table_name=f"docs_{ticker}",
        uri=db_path,
        embedder=embedder
    )

    # ------------------ Prepare Documents ------------------
    documents = []
    for text in texts:
        documents.append(
            Document(
                content=text,
                meta_data={
                    "ticker": ticker
                }
            )
        )

    # ------------------ Batch Insert ------------------
    vector_db.insert(
        content_hash=None,
        documents=documents
    )

    tqdm.write(f"   ✅ Inserted {len(documents)} pages into LanceDB")

    return Knowledge(
    vector_db=vector_db,
    max_results=5
    )


def search_duckduckgo(query: str, max_results=5) -> str:
    """Executes the search via Python."""
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No search results found for: {query}"
        output = []
        for article in results:
            title = article.get('title', 'N/A')
            url = article.get('href', 'N/A')
            body = article.get('body', 'N/A')
            output.append(f"Title: {title}\nSummary: {body}\nLink: {url}\n---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"

def get_market_analyst(knowledge_base=None):
    """Returns the Analyst Agent."""
    return Agent(
        model=Ollama(id=OLLAMA_MODEL_ID), 
        description="You are a cynical, forensic financial analyst.",
        instructions=[
            "You are investigating a stock for a potential investment.",
            "Your goal is to find the TRUTH, not just repeat the marketing hype.",
            "Analyze the provided Search Results (and Annual Report if available) to answer the user's request.",
            "NOISE FILTER: Disregard 'Shareholder Investigation' or 'Class Action' spam.",
            "Always cite your sources."
        ],
        knowledge=knowledge_base,
        search_knowledge=True if knowledge_base else False, 
        debug_mode=True,
        markdown=True,
    )

def safe_run(agent, initial_search_query, analysis_instruction, sleep_time=1, use_kb=False):
    """Cycle: Action (Search/RAG) -> Context -> Response."""
    current_query = initial_search_query
    accumulated_context = ""
    max_cycles = 2
    
    for cycle in range(max_cycles):
        if use_kb:
            tqdm.write(f"   📚 Cycle {cycle+1}: Querying Annual Report for '{current_query}'...")
            prompt = analysis_instruction
        else:
            tqdm.write(f"   🔎 Cycle {cycle+1}: Searching '{current_query}'...")
            new_results = search_duckduckgo(current_query)
            accumulated_context += f"\n\n=== SEARCH RESULTS (Round {cycle+1}) ===\n{new_results}\n==============================\n"
            
            prompt = f"""
            {analysis_instruction}
            {accumulated_context}
            You have gathered all available information. Please provide the final analysis now.
            """

        try:
            time.sleep(sleep_time)
            response = agent.run(prompt)
            if response and response.content:
                return response
        except Exception as e:
            tqdm.write(f"❌ Error during cycle {cycle+1}: {e}")
            return None
    return None

def analyze_sentiment_and_news(ticker):
    """Main Analyst Workflow."""
    print(f"\n🕵️‍♂️ Analyst Agent ({OLLAMA_MODEL_ID}): Starting Deep Dive for {ticker}...")
    
    company_name = get_company_name(ticker)
    start_time = time.time()

    # 1. Try to Download Annual Report
    download_annual_report(ticker, company_name)

    # 2. Initialize Knowledge Base
    kb = get_annual_report_kb(ticker)
    has_kb = True if kb else False
    
    agent = get_market_analyst(knowledge_base=kb)
    current_year = datetime.now().year
    
    research_notes = {
        "ANNUAL_REPORT_INSIGHTS": "No Annual Report found.", 
        "HISTORY": "No data found.",
        "RED_FLAGS": "No data found.",
        "RECENT": "No data found.",
        "MOAT": "No data found.",
        "MANAGEMENT": "No data found."
    }

    total_steps = 6 if has_kb else 5
    with tqdm(total=total_steps, desc="Initializing...", unit="step") as pbar:

        # --- STEP 0: Annual Report Audit (RAG) ---
        if has_kb:
            pbar.set_description("Step 0/5: Auditing Annual Report")
            resp_0 = safe_run(
                agent,
                initial_search_query="Risk Factors and Management Discussion",
                analysis_instruction="Query the Annual Report. Summarize the 'Risk Factors' and 'Management Discussion' sections. Highlight warnings about growth or litigation.",
                use_kb=True
            )
            if resp_0: research_notes["ANNUAL_REPORT_INSIGHTS"] = resp_0.content
            pbar.update(1)

        # --- STEP 1: Historical Context ---
        pbar.set_description("Step 1/5: Historical Context")
        resp_1 = safe_run(
            agent, 
            initial_search_query=f"{company_name} {ticker} corporate strategy events {current_year-5}..{current_year-2}",
            analysis_instruction=f"Summarize major corporate events for '{company_name}' between {current_year-5} and {current_year-2}."
        )
        if resp_1: research_notes["HISTORY"] = resp_1.content
        pbar.update(1)

        # --- STEP 2: Red Flags ---
        pbar.set_description("Step 2/5: Red Flags Check")
        resp_2 = safe_run(
            agent,
            initial_search_query=f"{company_name} lawsuit fraud SEC investigation short seller report",
            analysis_instruction=f"List any lawsuits, fraud allegations, or SEC investigations for '{company_name}'."
        )
        if resp_2: research_notes["RED_FLAGS"] = resp_2.content
        pbar.update(1)

        # --- STEP 3: Recent Drivers ---
        pbar.set_description("Step 3/5: Recent Drivers")
        resp_3 = safe_run(
            agent,
            initial_search_query=f"{company_name} earnings growth drivers stock news {current_year-2}..{current_year}",
            analysis_instruction=f"Summarize latest earnings results and growth drivers for '{company_name}'."
        )
        if resp_3: research_notes["RECENT"] = resp_3.content
        pbar.update(1)

        # --- STEP 4: Competitive Moat ---
        pbar.set_description("Step 4/5: Moat Analysis")
        resp_4 = safe_run(
            agent,
            initial_search_query=f"{company_name} competitive advantage moat network effect",
            analysis_instruction=f"Analyze the competitive advantages of '{company_name}'."
        )
        if resp_4: research_notes["MOAT"] = resp_4.content
        pbar.update(1)

        # --- STEP 5: Management Quality ---
        pbar.set_description("Step 5/5: Management Audit")
        resp_5 = safe_run(
            agent,
            initial_search_query=f"{company_name} management capital allocation insider ownership",
            analysis_instruction=f"Investigate the management of '{company_name}' (Capital Allocation, Insider Ownership)."
        )
        if resp_5: research_notes["MANAGEMENT"] = resp_5.content
        pbar.update(1)
        
        pbar.set_description("✅ Analysis Complete")

    # --- SAVE TO FILE ---
    end_time = time.time()
    formatted_time = str(timedelta(seconds=int(end_time - start_time)))
    
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    safe_ticker = ticker.replace(".", "_")
    filename = f"research_{safe_ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"ANALYST RESEARCH REPORT FOR {company_name} ({ticker})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        for k, v in research_notes.items():
            f.write(f"[{k}]\n{v}\n\n{'-'*30}\n\n")
            
    print(f"✅ Research saved: {filepath}")
    return filepath

if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. NVDA): ").strip()
    analyze_sentiment_and_news(ticker)