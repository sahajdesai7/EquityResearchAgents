# PEVC Dealbox Agent

This agent automates the sourcing and screening of business listings from **Smergers**. It crawls raw data, normalizes financials using an LLM (Groq), and filters candidates based on specific Buy-Side criteria (e.g., EBITDA margins, Deal Size, and Geography).

## 🏗️ Technical Architecture
The agent operates as a sequential pipeline consisting of three distinct layers:

### 1. Ingestion Layer (Crawler)
**Tool: Firecrawl API**

Function: Converts unstructured HTML from Smergers into an LLM-ready structured schema.

Output: Raw JSON containing listings with generic headlines and dirty financial strings.

### 2. Intelligence Layer (Cleaner)
**Tool: Groq API (openai/gpt-oss-120b) + Pydantic**

Function:

Parsing: Extracts structured fields (City, Country, Currency, Units) from raw text.

Math Engine: Python-based deterministic calculation (not LLM hallucination) handles currency conversion (e.g., INR Crores to EUR Millions) and margin averaging.

Output: Clean, normalized JSON with standard units (EUR Millions).

### 3. Decision Layer (Filter)
**Tool: Python Logic + Groq Semantic Check**

Function:

Hard Filter: Applies strict financial thresholds (e.g., Sales > €1M for full sales).

Semantic Filter: Uses Groq to validate geography (e.g., "Is 'Bangkok' in Asia?").

Output: Final "Deal Box" candidates ready for analyst review.

## 🚀 Setup & Usage
Prerequisites: Python 3.9+

### 1. Install Dependencies:

Bash: pip install -r requirements.txt

### 2. Configure Environment: 

Create a .env file with your keys:

FIRECRAWL_API_KEY=fc-your_key
GROQ_API_KEY=gsk_your_key
GROQ_MODEL_ID=openai/gpt-oss-120b

### 3. Run Pipeline:

Bash: python main.py

## 📂 Folder Structure

```text
PEVC-dealbox-agent/
├── .env                    # API Keys (Firecrawl, Groq)
├── main.py                 # Master script to run the pipeline
├── requirements.txt        # Python dependencies
├── json/                   # Stores raw and processed data
│   ├── smergers_data.json
│   ├── smergers_data_genai_cleaned.json
│   └── dealbox_candidates.json
└── modules/                # Agent Logic
    ├── crawl_smergers.py        # Extracts raw data
    ├── clean_smergers_llm.py    # Standardizes data via Groq
    └── filter_dealbox.py        # Applies investment thesis
```

## 📄 License
The MIT License (MIT)

Copyright (c) 2024 PEVC Dealbox Agent

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.