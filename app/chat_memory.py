from typing import List, Dict
import time, uuid

class ChatMemory:
    def __init__(self):
        self.store = {}

    def create_session(self):
        sid = str(uuid.uuid4())
        self.store[sid] = []
        return sid

    def add_message(self, session_id: str, role: str, text: str):
        if session_id not in self.store:
            self.store[session_id] = []
        self.store[session_id].append({"role": role, "text": text, "time": time.time()})

    def get_history(self, session_id: str, last_n: int = 10):
        return self.store.get(session_id, [])[-last_n:]
