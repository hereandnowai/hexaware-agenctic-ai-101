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