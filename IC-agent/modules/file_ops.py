import os
import json
from typing import List, Dict

def read_prompt(file_path: str) -> str:
    """
    Reads a text file and returns the content as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return ""

def save_chat_history(directory: str, filename: str, messages: List[Dict[str, str]]):
    """
    Saves the list of message dictionaries to a JSON file.
    Creates the directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    full_path = os.path.join(directory, filename)
    
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=4)
        print(f"Chat history saved to {full_path}")
    except Exception as e:
        print(f"Error saving chat history: {e}")

def get_latest_user_input(messages: List[Dict[str, str]]) -> str:
    """
    Helper to extract the last message sent by the user.
    """
    for msg in reversed(messages):
        if msg['role'] == 'user':
            return msg['content']
    return ""