from typing import Optional
from openai import OpenAI
import base64
import os
import re
import time
import warnings
import logging
from FRM.file_process import file_to_string
import json
import threading

def make_client(role='llm'):
    """OpenAI-compatible client. VLM_* / LLM_* environment variables (key, base URL)
    fall back to the original MY_API_KEY and OPENAI_BASE_URL."""
    prefix = 'VLM' if role == 'vlm' else 'LLM'
    key = os.environ.get(f'{prefix}_API_KEY') or os.environ.get('MY_API_KEY')
    base_url = os.environ.get(f'{prefix}_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
    if not key:
        raise RuntimeError(f"No API key for the {prefix}: set {prefix}_API_KEY (or MY_API_KEY)")
    return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)


# Optional request pacing for rate-limited (e.g. free-tier) keys: VLM_RPM / LLM_RPM = max requests per minute.
_pace_lock = threading.Lock()
_next_slot = {}


def pace(role):
    rpm = float(os.environ.get('VLM_RPM' if role == 'vlm' else 'LLM_RPM', 0) or 0)
    if rpm <= 0:
        return
    with _pace_lock:
        now = time.time()
        slot = max(now, _next_slot.get(role, 0.0))
        _next_slot[role] = slot + 60.0 / rpm
    if slot > now:
        time.sleep(slot - now)


def is_rate_limit(e):
    msg = str(e).lower()
    return getattr(e, 'status_code', None) == 429 or any(k in msg for k in ('429', 'rate limit', 'quota', 'resource_exhausted', 'too many requests'))

# for create a dialogue
class Conversation():
    def __init__(self, system_prompt: Optional[str]) -> None:
        if system_prompt is None:
            self.messages = []
        else:
            self.messages = [
                {
                "role": "system",
                "content": system_prompt
                }
            ]
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0

    def add_system_prompt(self, system_prompt:str) -> None:
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })

    def add_assistant_content(self, assistant_content:str):
        self.messages.append(
            {
            "role": "assistant",
            "content": assistant_content
            }
        )

    def add_user_content(self, contents:list):
        """
        contents format:
        [
            {"type": "text", "data": ""}, 
            {}
            ]
        """
        full_content = []
        for content in contents:
            if content["type"] == "text":
                 full_content.append(
                    {
                    "type": "text", 
                    "text": content["data"]
                    }
                )
            elif content["type"] == "image_url":
                full_content.append(
                    {
                    "type": "image_url",
                    "image_url": 
                        {
                        "url": f"data:image/png;base64,{content['data']}",
                        "detail": "high"
                        }
                    }
                )
            else:
                raise ValueError("content type error. ")
        
        self.messages.append(
            {
            "role": "user",
            "content": full_content
            }
        )
    
    def update_usage(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

    def clear_prompt(self):
        # clear prompt without system prompt
        self.messages.clear()



class Agent():
    def __init__(self, system_prompt, model_type="gpt-4.1-mini", temperature=1, role='llm'):
        
        self.role = role
        self.client = make_client(role)
        self.model_type = model_type
        print(f"Using LLM: {model_type}")
        self.conversation = Conversation(system_prompt)
        self.temperature = temperature

    def query(self, sample_num, once_sample_num):
        responses = []
        response_cur = None
        total_samples = 0

        while True:
            if total_samples >= sample_num:
                break
            for attempt in range(20):
                try:
                    pace(self.role)
                    response_cur = self.client.chat.completions.create(
                        model=self.model_type,
                        messages=self.conversation.messages,
                        temperature=self.temperature,
                        n=once_sample_num
                    )
                    total_samples += once_sample_num
                    break
                except Exception as e:
                    if attempt >= 10:
                        # some OpenAI-compatible APIs reject n > 1: fall back to fewer samples per call
                        once_sample_num = 1 if attempt >= 15 else max(int(sample_num / 2), 1)
                        print("Current once sample num", once_sample_num)
                    print(f"Failed with error: {e}\n{self.model_type} attempt {attempt + 1}")
                    # rate limits: back off (5 s doubling up to 2 min) instead of failing fast
                    time.sleep(min(120, 5 * 2 ** min(attempt, 5)) if is_rate_limit(e) else 2)
            if response_cur is None:
                print("Code terminated due to too many failed attempts! "
                      "(If these were rate-limit errors, the API key's daily quota may be used up.)")
                exit()

            responses.extend(response_cur.choices)
            if getattr(response_cur, 'usage', None) is not None:
                self.conversation.update_usage(response_cur.usage)

        return responses



