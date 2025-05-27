import re
import shutil
import subprocess
import sys
import tempfile
import os

from camel.messages import BaseMessage
from pipeline import answer_vlsi_query, camel_agent

# shared namespace for exec‐fallback (if needed)
shared_globals = {}


def extract_python_code(message: str) -> str:
    """
    Return the first ```python … ``` fence's contents, or None.
    """
    m = re.search(r"```python(.*?)```", message, re.DOTALL)
    return m.group(1).strip() if m else None


def execute_code(
    code: str,
    context: dict = None,
    use_openroad: bool = True
) -> str:
    """
    If `import openroad` is found and openroad is on PATH → run
        openroad -python <tempfile>
    else → run via sys.executable.
    Returns combined stdout+stderr plus debug info.
    """
    # dump to temp file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    # pick runner
    if use_openroad and "import openroad" in code and shutil.which("openroad"):
        runner = ["openroad", "-python", path]
    else:
        runner = [sys.executable, path]

    proc = subprocess.run(runner, capture_output=True, text=True)
    try:
        os.remove(path)
    except OSError:
        pass

    # debug string
    parts = [
        f"Command: {' '.join(runner)}",
        f"Exit code: {proc.returncode}",
        "----- STDOUT -----",
        proc.stdout or "<empty>",
        "----- STDERR -----",
        proc.stderr or "<empty>",
    ]
    return "\n".join(parts)


def run_query_and_execute(
    user_query: str,
    top_k: int = 7,
    similarity_threshold: float = 0.2
) -> str:
    """
    1) Calls answer_vlsi_query() → may emit a ```python``` block
    2) Extracts the code, prints it
    3) Runs it under openroad (or fallback), prints debug
    4) Feeds the log back to the LLM as a follow‐up
    5) Returns the LLM's final reply
    """
    # 1) get initial LLM reply
    first = answer_vlsi_query(
        query=user_query,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

    # 2) extract code
    code = extract_python_code(first)
    if not code:
        return first

    # DEBUG print
    print("=== Extracted Python code ===\n", code, "\n=== End code ===\n")

    # 3) execute
    out = execute_code(code, shared_globals, use_openroad=True)

    # DEBUG print
    print("=== Execution output ===\n", out, "\n=== End output ===\n")

    # 4) send back to LLM
    followup = BaseMessage.make_user_message(
        role_name="Executor",
        content=f"I ran your Python block under `openroad -python` and got:\n```\n{out}\n```"
    )
    resp = camel_agent.step(followup)

    # 5) return final LLM reply
    return "\n".join(m.content for m in resp.msgs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run a VLSI/OpenROAD query end-to-end (LLM → OpenROAD → LLM)"
    )
    parser.add_argument("query", help="Your VLSI or OpenROAD question")
    parser.add_argument("--top_k", type=int, default=7)
    parser.add_argument("--sim_thresh", type=float, default=0.2)
    args = parser.parse_args()

    result = run_query_and_execute(
        user_query=args.query,
        top_k=args.top_k,
        similarity_threshold=args.sim_thresh
    )
    print(result) 