from typing import TypedDict, List, Annotated, Dict, Union
import operator
from langchain_core.messages import BaseMessage

# Define the structure of our State
class AgentState(TypedDict):
    # We use BaseMessage to support HumanMessage/AIMessage objects
    messages: Annotated[List[BaseMessage], operator.add] 
    
    question_count: int
    
    # Track confidence per agent (0.0 to 1.0)
    agent_confidence: Dict[str, float] 
    
    # Who spoke last?
    last_speaker: str
    
    # The final verdict
    verdict: str