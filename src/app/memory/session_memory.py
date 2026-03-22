"""
session_memory.py

Simple in-memory session-based chat history store.

Maps:
    session_id -> list[message dict]

Message format:
    {
        "role": "user" | "assistant",
        "content": str
    }
"""

from collections import defaultdict
from typing import List, Dict


class SessionMemory:

    def __init__(self, max_messages: int = 10):
        """
        Args:
            max_messages (int):
                Maximum messages stored per session.
        """
        self._store = defaultdict(list)
        self.max_messages = max_messages

    def get_history(self, session_id: str) -> List[Dict]:

        return self._store[session_id]

    def add_user_message(self, session_id: str, content: str):

        self._store[session_id].append(
            {"role": "user", "content": content}
        )

        self._trim(session_id)

    def add_ai_message(self, session_id: str, content: str):

        self._store[session_id].append(
            {"role": "assistant", "content": content}
        )

        self._trim(session_id)

    def _trim(self, session_id: str):

        history = self._store[session_id]

        if len(history) > self.max_messages:

            self._store[session_id] = history[-self.max_messages:]