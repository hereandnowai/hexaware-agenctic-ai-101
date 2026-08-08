# What is HARNESS?
# A pre-assembled agent for long, multi-step jobs.

import inspect
import shutil
from pathlib import Path
from typing import Annotated
from agent_framework import (FileSystemAgentFileStore,
                             FunctionInvocationContext,
                             create_harness_agent, tool)

from _maf import POLICY, banner, get_client, run

banner("File - 12 - The Harness")

OPTIONS = list(inspect.signature(create_harness_agent).parameters)
print(f" create_harness_agent() takes {len(OPTIONS)} arguments")

GROUPS = {
    "plans its own work": ["disable_todo", "todo_provider"],
    "survives a long run": ["max_context_window_tokens", "disable-compaction"],
    "read and write files": ["file_access_store", "disable_file_access"],
    "remembers across runs": ["diable_file_memory", "file_memory_store"],
    "reaches outware": ["disable_web_search", "shell_executor", "background_agents"],
    "pre-wired": ["disable_tool_auto_approval", "auto_approval_rules"]
}
for group, names in GROUPS.items():
    print(f" {group} {', '.join(n for n in names if n in OPTIONS)}")
print()

QUEUE = {
    "HX-90455": {"days": 12, "faulty": True, "item": "Hex Studio headphones", "paid": 129.99},
    "HX-90456": {"days": 34, "faulty": False, "item": "Hex Buds Mk II", "paid": 59.00},
    "HX-90457": {"days": 6, "faulty": False, "item": "Hex Studio headphones", "paid": 129.00},
    "HX-90458": {"days": 27, "faulty": True, "item": "Hex Desk Mic", "paid": 84.50}
}

CALLS: list[str] = []
WORKSPACE = Path(__file__).parent / "_harness_workspace"

async def watch(context: FunctionInvocationContext, next):
    """Record the name of every tool that is called, in order."""
    CALLS.append(context.function.name)
    await next()

@tool
def list_open_returns() -> str:
    """Every return request currently waiting for triage."""
    return ', '.join(QUEUE)

@tool
def lookup_return(order_id: Annotated[str, "an order reference"]) -> str:
    """Read a return request. Changes nothing"""
    record = QUEUE.get(order_id)
    if not record:
        return f"{order_id}: not found"
    return (f"{order_id}: {record['item']}, USD {record['paid']}, delivered"
            f"{record['days']} days ago, faulty={record['faulty']}")

TASK = (
    "Triage every open return. For each one decide ACCEPT or REJECT " \
    "under our policy and give a short reason. Then write the whole thing to " \
    "triage_report.md as a markdown table with columns order_id, decision, reason. " \
    "Finally, write a summary to the console with USD we are refunding."
)

TOOLS = [list_open_returns, lookup_return]

async def main():
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True)

    # Part A - a plain agent with no equipment
    plain = get_client().as_agent(
        name="triage",
        tools=TOOLS,
        middleware=[watch],
        instructions=f"You are Hex Retail's return desk. {POLICY}"),
    answer_a = await plain.run(TASK)
    calls_a, CALLS[:] = list(CALLS), []
    files_a = list(WORKSPACE.iterdir())

    # Part B - The Harness, same job, same tools
    harness = create_harness_agent(
        client=get_client(),
        name="triage",
        tools=TOOLS,
        middleware=[watch],
        agent_instructions=f"You are Hex Retail's return desk. {POLICY}",
        file_access_store=FileSystemAgentFileStore(root_directory=str(WORKSPACE)),
        file_access_disable_write_tool_approval=True,
        disable_web_search=True,
        disable_file_memory=True,
        disable_mode=True,
        loop_max_iterations=10)

    session = harness.create_session()
    await harness.run(TASK, session=session)
    calls_b, files_b = list(CALLS), list(WORKSPACE.iterdir())

    # The difference
    mine = {t.name for t in TOOLS}
    borrowed = [c for cin calls_b if c not in mine]
    todo = session.state.get("todo". {}.get('items', []))

    print(f" calls with plain agent: {calls_a}")
    print(f" calls with harness agent: {calls_b}")
    print(f" files with plain agent: {files_a}")
    print(f" files with harness agent: {files_b}")
