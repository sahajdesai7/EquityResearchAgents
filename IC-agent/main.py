import sys
import random
import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from modules.state_manager import AgentState
from modules.file_ops import read_prompt, save_chat_history
# Import the new analysis function
from modules.agent_brain import visionary, fundamentalist, pragmatist, get_verdict, analyze_opening_pitch

# --- CONFIGURATION ---
VISIONARY_PROMPT = read_prompt("prompts/visionary_agent.txt")
FUNDAMENTALIST_PROMPT = read_prompt("prompts/fundamentalist_agent.txt")
PRAGMATIST_PROMPT = read_prompt("prompts/pragmatist_agent.txt")
ORCHESTRATOR_PROMPT = read_prompt("prompts/orchestrator.txt")

# --- THE ORCHESTRATOR ---
def orchestrator_logic(state: AgentState) -> Literal["visionary", "fundamentalist", "pragmatist", "verdict"]:
    # 1. STOP CONDITION CHECK
    current_confidences = state.get("agent_confidence", {}).values()
    avg_confidence = sum(current_confidences) / 3 if current_confidences else 0.0
    question_count = state.get("question_count", 0)
    
    print(f"\n[DEBUG] Question: {question_count}/5 | Avg Confidence: {avg_confidence:.2f}")

    if question_count >= 5 or avg_confidence >= 0.8:
        return "verdict"

    # 2. DYNAMIC ROUTING
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""
    
    if any(word in last_message for word in visionary.triggers['green']):
        target = "visionary"
    elif any(word in last_message for word in fundamentalist.triggers['green']):
        target = "fundamentalist"
    elif any(word in last_message for word in pragmatist.triggers['green']):
        target = "pragmatist"
    else:
        options = ["visionary", "fundamentalist", "pragmatist"]
        last_speaker = state.get("last_speaker", "")
        if last_speaker in options:
            options.remove(last_speaker)
        target = random.choice(options)
        
    print(f"[DEBUG] Routing to: {target.upper()}")
    return target

# --- NODES ---
def call_visionary(state: AgentState):
    return visionary.evaluate(state, VISIONARY_PROMPT)

def call_fundamentalist(state: AgentState):
    return fundamentalist.evaluate(state, FUNDAMENTALIST_PROMPT)

def call_pragmatist(state: AgentState):
    return pragmatist.evaluate(state, PRAGMATIST_PROMPT)

def call_verdict(state: AgentState):
    return get_verdict(state)

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("visionary", call_visionary)
workflow.add_node("fundamentalist", call_fundamentalist)
workflow.add_node("pragmatist", call_pragmatist)
workflow.add_node("verdict", call_verdict)

workflow.set_conditional_entry_point(
    orchestrator_logic,
    {"visionary": "visionary", "fundamentalist": "fundamentalist", "pragmatist": "pragmatist", "verdict": "verdict"}
)

workflow.add_edge("visionary", END)
workflow.add_edge("fundamentalist", END)
workflow.add_edge("pragmatist", END)
workflow.add_edge("verdict", END)

app = workflow.compile()

# --- CLI RUNNER ---
def run_cli():
    print("---------------------------------------------------------")
    print("🦈 INVESTMENT COMMITTEE SIMULATION (CLI) 🦈")
    print("---------------------------------------------------------")
    print("Type 'quit' to exit.")
    
    # 1. GET OPENING PITCH
    pitch_input = input("\nDeveloped an idea? Pitch it > ")
    if pitch_input.lower() in ["quit", "exit"]:
        return

    # 2. ANALYZE PITCH (Set Initial Confidence)
    initial_scores = analyze_opening_pitch(pitch_input)
    
    # 3. INITIALIZE STATE
    initial_state = {
        "messages": [HumanMessage(content=pitch_input)],
        "question_count": 0,
        "agent_confidence": initial_scores, # Use the calculated scores
        "last_speaker": None,
        "verdict": None
    }
    
    # 4. START LOOP
    while True:
        # Run Graph
        result = app.invoke(initial_state)
        initial_state = result
        
        last_msg = initial_state["messages"][-1]
        print(f"\n{last_msg.content}")
        
        if initial_state.get("verdict"):
            print("\n---------------------------------------------------------")
            print(f"FINAL VERDICT: {initial_state['verdict']}")
            print("---------------------------------------------------------")
            save_chat_history("chat_history", "session.json", [m.dict() for m in initial_state["messages"]])
            break
            
        # Get next user input
        user_input = input("\nYour Answer > ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        initial_state["messages"].append(HumanMessage(content=user_input))

if __name__ == "__main__":
    run_cli()