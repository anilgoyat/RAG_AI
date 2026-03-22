"""
llm_service.py — Step 6: LLM Call

Responsibility:
    Send a list of messages to an OpenAI LLM via LangChain ChatOpenAI
    and return the generated text response.

Model:
    gpt-4o-mini (default — configurable via config.py)

Input  : messages (list[dict])
Output : str
"""

import logging
import sys
from pathlib import Path
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    OPENAI_API_KEY,
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _to_langchain_messages(messages: list[dict]) -> list:
    """
    Convert OpenAI-style dict messages into LangChain message objects.
    """

    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    converted = []

    for msg in messages:

        role = msg.get("role")

        content = msg.get("content", "")

        if role not in role_map:

            raise ValueError(
                f"Unsupported message role: '{role}'. "
                "Use system/user/assistant."
            )

        converted.append(
            role_map[role](content=content)
        )

    return converted


# ──────────────────────────────────────────────
# LLM Service
# ──────────────────────────────────────────────

class LLMService:
    """
    Wrapper around LangChain ChatOpenAI client.

    Loads model once during initialization.
    Accepts OpenAI-style messages and returns text response.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):

        if not OPENAI_API_KEY:

            raise ValueError(
                "OPENAI_API_KEY missing. "
                "Set it in config.py or .env"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            api_key=SecretStr(OPENAI_API_KEY),
        )

        logger.info(
            f"LLMService initialized — model='{self.model}', "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )


    # ──────────────────────────────────────────
    # Generate response
    # ──────────────────────────────────────────

    def generate(
        self,
        messages: list[dict]
    ) -> str:

        if not messages:

            raise ValueError(
                "messages list is empty — nothing to send to the LLM."
            )

        logger.info(
            f"Calling OpenAI model '{self.model}' "
            f"with {len(messages)} message(s)..."
        )

        try:

            lc_messages = _to_langchain_messages(
                messages
            )

            response = self._client.invoke(
                lc_messages
            )

            reply = response.content

            if isinstance(reply, list):
             reply = " ".join(
                item if isinstance(item, str) else str(item)
                for item in reply
            )

            logger.info(
                f"Response received — {len(reply)} character(s)."
            )

            return reply

        except Exception as e:

            raise RuntimeError(
                f"OpenAI API call failed for model '{self.model}': {e}"
            ) from e


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    llm = LLMService()

    messages = [

        {
            "role": "system",
            "content": "You are a helpful assistant. Answer concisely."
        },

        {
            "role": "user",
            "content": "What is a transformer model in machine learning? Answer in 2 sentences."
        },
    ]

    print(f"\nModel  : {llm.model}")

    print(f"Prompt : {messages[-1]['content']}")

    print("-" * 50)

    response = llm.generate(messages)

    print(f"Response:\n{response}")