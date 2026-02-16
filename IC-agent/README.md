# Investment Committee (IC) Simulation Agent

This agent simulates a high-stakes Investment Committee meeting, allowing founders to practice their startup pitches against AI personas modeled after *Shark Tank India* investors. It operates as a CLI (Command Line Interface) tool where a "Founder" pitches to a panel of three distinct AI agents, managed by an intelligent orchestrator.

## 🏗️ Technical Architecture

The system utilizes **LangGraph** to manage state and **Groq** for high-speed LLM inference. The logic is divided into three core layers:

### 1. The Orchestration Layer (The Router)
**Tool: LangGraph State Machine**

**Function**: Acts as the moderator of the meeting.
* **Dynamic Routing**: Analyzes the user's input for specific "trigger words" (e.g., "revenue" triggers the Fundamentalist, "brand" triggers the Visionary).
* **Stop Condition Logic**: Automatically ends the session if:
    * The "Question Limit" (5 questions) is reached.
    * The "Average Confidence Score" of the committee exceeds 80%.

### 2. The Cognitive Layer (The Brains)
**Tool: Groq API (via LangChain)**

**Function**: Simulates three distinct investor personas:
* **🦈 The Visionary (Agent A)**: Focuses on Brand, Story, and "Spark" (Marketing-driven).
* **🦈 The Fundamentalist (Agent B)**: Focuses on Unit Economics, Margins, and Scale (Math-driven).
* **🦈 The Pragmatist (Agent C)**: Focuses on Operations, Team, and Execution (Risk-averse).

**Advanced Logic Features**:
* **Opening Pitch Analysis**: Before the loop starts, the system analyzes the initial pitch to set a dynamic baseline confidence score (e.g., a weak pitch starts at 0.1, a strong one at 0.4).
* **Global Confidence Updates**: After *every* user answer, the system calculates the "Confidence Impact" for **all three agents** simultaneously, ensuring the silent agents still react to the founder's answers.
* **Smart Judge**: An internal classifier checks if the Agent's response was a *question* or just a *remark*. The question counter only increments for actual interrogations.

### 3. The Data Layer (State & Persistence)
**Tool: Python (`file_ops.py`, `state_manager.py`)**

**Function**:
* **State Management**: Tracks conversation history, individual agent confidence scores, and turn counts.
* **Persistence**: Automatically saves the full transcript to `chat_history/session.json` upon completion.

---

## 🧠 Logic Flow

### 1. Ingestion:
User enters Opening Pitch.

### 2. Analysis: 
System calculates initial confidence for all agents.

### 3. Loop:
- Orchestrator checks Stop Conditions -> Routes to Agent. <br>
- Agent analyzes user input -> Updates Global Confidence -> Generates Response.<br>
- Judge classifies response -> Updates Question Count.<br>
- User responds.

### 4. Verdict: 
Once the loop breaks, the committee outputs a final **INVEST** or **PASS** decision with reasoning.

## 🚀 Setup & Usage

**Prerequisites**: Python 3.9+

### 1. Install Dependencies
bash 
```bash
pip install -r requirements.txt
```
### 2. Configure Environment: 

Create a .env file with your keys:
```text
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL_ID=openai/gpt-oss-120b
```
### 3. Run Pipeline:
bash 
```bash
python main.py
```
## 📂 Folder Structure

```text
IC-agent/
├── .env                    # API Keys (Groq)
├── main.py                 # Entry point: Orchestrator & CLI Loop
├── requirements.txt        # Dependencies (LangChain, LangGraph, Dotenv)
├── chat_history/           # Saved sessions (JSON)
│   └── session.json
├── modules/                # Core Logic
│   ├── agent_brain.py      # LLM Personas, Sentiment Analysis, Smart Judge
│   ├── state_manager.py    # LangGraph State Schema (TypedDict)
│   └── file_ops.py         # File reading/writing utilities
└── prompts/                # System Prompts (Personality Definitions)
    ├── visionary_agent.txt
    ├── fundamentalist_agent.txt
    ├── pragmatist_agent.txt
    └── orchestrator.txt
```

## 📄 License
The MIT License (MIT)

Copyright (c) 2026 Investment Committee Agent

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.