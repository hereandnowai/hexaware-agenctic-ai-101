import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, Message
from agent_framework.observability import enable_instrumentation
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes
from pydantic import Field

load_dotenv(Path(__file__).parent / ".env")
MODEL = os.environ["GOOGLE_MODEL"]
API_KEY = os.environ["GOOGLE_API_KEY"]
BASE_URL = os.environ["GOOGLE_BASE_URL"]

langfuse = Langfuse()
enable_instrumentation(enable_sensitive_data=True)

ACCOUNTS = {"SB-9001": 84_215.50, "SB-9002": 12_430.00, "SB-9003": 3_46_890.25}

def check_balance(account_id: Annotated[str, Field(description="Account id, e.g. SB-9001")]) -> str:
    """Look up the balance of a Meridian Bank Account."""
    balance = ACCOUNTS.get(account_id.upper())
    return f"{account_id}: Rs{balance:,.2f}" if balance else f"No account {account_id}"

agent = Agent(
    OpenAIChatCompletionClient(model=MODEL, api_key=API_KEY, base_url=BASE_URL),
    "You are Meridian Bank's Acssistant capable of checking the Account Balance of a customer. "
    "You will be given an account id, and you will respond with the balance of that account." \
    "If the account id is not found, you will respond with 'No account {account_id}'."
    "Branches open Monday to Friday, 9am to 5pm. Saturday 9am to 1pm. Closed on Sunday and public holidays."
    "Savings pays 3% interest per year, and Fixed Deposits pay 6% interest per year. "
    "You will not provide any other information, and you will not make up any information." \
    "Be brief and concise in your responses.",
    name="Meridian Bank Assistant",
    tools=[check_balance]
)

THOUGHTS = re.compile(r"<thought>.*?</thought>|<thought>.*", re.DOTALL)

def strip_thoughts(text: str) -> str:
    """Drop Gemma's thinking and leave only the answer"""
    return THOUGHTS.sub("", text).replace("</thought>", "").strip()

def plain_text(content: object) -> str:
    """Flatten one Gradio history"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(plain_text(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)

def new_session() -> dict:
    """Fresh, empty running totals for one conversation"""
    return {"id": f"chat-{uuid.uuid4().hex[:12]}", "turns": 0,
            "tokens_in": 0, "tokens_out": 0, "tokens_think": 0, "seconds":0.0}

def trace_links(trace_id: str, session_id: str) -> str:
    """Two markdown links: this one turn, and the whole conversation it belongs to."""
    turn = langfuse.get_trace_url(trace_id=trace_id) or ""
    whole = f"{turn.rsplit('/traces/', 1)[0]}/sessions/{session_id}"
    return f"[This turn's trace]({turn}) | [The whole conversation]({whole})"

def ask(message: str, history: list, session: dict | None = None) -> tuple[str, str]:
    session = new_session() if session is None else session
    past = [Message(m["role"], [plain_text(m["content"])]) for m in history]
    started = time.perf_counter()

    with propagate_attributes(session_id=session["id"]), \
                langfuse.start_as_current_observation(name="chat turn", as_type="agent") as span:
            trace_id = span.trace_id
            try:
                result = asyncio.run(agent.run([*past, Message("user", [message])]))
            except Exception as error:
                span.update(level="ERROR", status_message=str(error))
                langfuse.flush()
                return (f"That turn failed: {type(error).__name__}",
                        f"Error {error}\n\n"
                        f"({langfuse.get_trace_url(trace_id=trace_id)})")

    elapsed = time.perf_counter() - started
    langfuse.flush()
    
    used = result.usage_details or {}
    token_in = used.get("input_token_count") or 0
    tokens_out = used.get("output_token_count") or 0
    tokens_think = max((used.get("total_token_count") or 0) - token_in - tokens_out, 0)

    session["turns"] += 1
    session["tokens_in"] += token_in
    session["tokens_out"] += tokens_out
    session["tokens_think"] += tokens_think
    session["seconds"] += elapsed

    report = (f"**This turn** | **{token_in}** in | **{tokens_out:,}** out | "
              f"**{tokens_think:,}** thinking | **{elapsed:.2f}s**\n\n"
              f"{trace_links(trace_id, session['id'])}")
    return strip_thoughts(result.text), report

if __name__ == "__main__":
    session, history = new_session(), []
    for question in ["What's the balance on SB-9001?", "And SB-9003"]:
        answer, summary = ask(question, history, session)
        history += [{"role": "user", "content": question,
                    "role": "assistant", "content": answer}]
        print(f"\n {question}\n{answer}\n\n{summary}")