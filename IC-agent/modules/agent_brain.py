import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from modules.state_manager import AgentState

# --- CONFIGURATION ---
# Define the profiles here so we can iterate over them for global updates
AGENT_PROFILES = [
    {"name": "Visionary", "role": "Brand & Marketing Expert (Emotional)"},
    {"name": "Fundamentalist", "role": "Skeptical Math & Finance Expert (Logical)"},
    {"name": "Pragmatist", "role": "Operations & Execution Expert (Practical)"}
]

# Initialize LLMs
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL_ID"), 
    temperature=0.7
)

judge_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL_ID"),
    temperature=0.0
)

def classify_intent(text: str) -> bool:
    """Check if the agent asked a question."""
    prompt = f"Analyze this statement. Does it contain a direct question? Statement: '{text}'. Respond YES or NO."
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    return "YES" in response.content.strip().upper()

def calculate_confidence_impact(user_input: str, agent_role: str, is_opening: bool = False) -> float:
    """
    Analyzes the pitch/answer quality.
    
    Calibration Adjustment:
    - We shifted the range to be slightly more forgiving.
    - Good answers should consistently yield +0.05 to +0.15.
    - Only bad answers should yield negative scores.
    """
    context_type = "OPENING PITCH" if is_opening else "FOUNDER ANSWER"
    
    prompt = f"""
    You are a critical Investment Judge acting as a '{agent_role}'.
    
    Analyze the {context_type}: "{user_input}"
    
    Score impact on confidence (-0.15 to +0.20):
    - +0.10 to +0.20: Excellent, specific, highly relevant to your role.
    - +0.01 to +0.09: Decent, reasonable, acceptable.
    - 0.00: Neutral or irrelevant.
    - -0.01 to -0.15: Vague, evasive, unrealistic, or 'red flag'.
    
    Output ONLY a single floating-point number. Example: 0.05
    """
    try:
        response = judge_llm.invoke([HumanMessage(content=prompt)])
        score = float(response.content.strip())
        # Clamp to ensure safety
        return max(-0.15, min(0.20, score))
    except ValueError:
        return 0.0

def analyze_opening_pitch(pitch_text: str):
    """
    Calculates the starting confidence for ALL agents based on the first pitch.
    """
    print("\n[System] Analyzing Opening Pitch... ⏳")
    initial_scores = {}
    
    for profile in AGENT_PROFILES:
        score = calculate_confidence_impact(pitch_text, profile["role"], is_opening=True)
        # Base confidence starts at 0.1, plus the impact of the pitch
        # e.g., Great pitch (+0.2) -> Starts at 0.3
        start_conf = max(0.1, min(0.5, 0.1 + score)) 
        initial_scores[profile["name"]] = start_conf
        print(f"   [Start]: {profile['name']} initialized at {start_conf:.2f}")
        
    return initial_scores

class InvestmentAgent:
    def __init__(self, name, role, green_lights, kill_switches):
        self.name = name
        self.role = role
        self.triggers = {"green": green_lights, "kill": kill_switches}

    def evaluate(self, state: AgentState, system_prompt: str):
        # 1. Get User's Latest Input
        latest_user_input = state["messages"][-1].content
        
        # 2. GLOBAL CONFIDENCE UPDATE (All Agents React)
        current_confidence_map = state["agent_confidence"].copy()
        
        print(f"\n   [Judge] Analyzing impact on ALL Sharks:")
        for profile in AGENT_PROFILES:
            # Calculate impact for this specific profile
            impact = calculate_confidence_impact(latest_user_input, profile["role"])
            
            # Update their score in the map
            curr = current_confidence_map.get(profile["name"], 0.1)
            new_val = max(0.0, min(1.0, curr + impact))
            current_confidence_map[profile["name"]] = new_val
            
            # Visual feedback
            sign = "+" if impact >= 0 else ""
            print(f"     -> {profile['name']}: {sign}{impact:.2f} (New: {new_val:.2f})")

        # 3. Generate Verbal Response (Only the Active Agent speaks)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response_msg = llm.invoke(messages)
        content = response_msg.content
        
        # 4. Question Counter
        is_question = classify_intent(content)
        increment = 1 if is_question else 0
        
        return {
            "messages": [response_msg],
            "agent_confidence": current_confidence_map, # Returns the GLOBALLY updated map
            "last_speaker": self.name,
            "question_count": state["question_count"] + increment
        }

def get_verdict(state: AgentState):
    final_prompt = "Review the conversation. Based on the confidence scores, output 'INVEST' or 'PASS' with a brief reason."
    messages = state["messages"] + [HumanMessage(content=final_prompt)]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "verdict": response.content
    }

# Instantiate Agents
visionary = InvestmentAgent("Visionary", "Brand Expert", ["brand", "story", "design"], ["copycat"])
fundamentalist = InvestmentAgent("Fundamentalist", "Skeptical Math Expert", ["profit", "margin", "scale"], ["burn"])
pragmatist = InvestmentAgent("Pragmatist", "Operational Expert", ["clean", "team", "focus"], ["messy"])