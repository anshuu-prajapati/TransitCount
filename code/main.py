"""
Agentic SDLC Pipeline - "Hello Rushikesh" UI Generator
=========================================================
6 agents that talk to each other automatically. You just run:
    python main.py

Pipeline:
  Agent 1 (Requirement Planner)  -> writes requirements.md
  Agent 2 (System Architect)     -> reads requirements.md, writes document.md
  Agent 3 (QA Auditor)           -> scores spec, loops back to Agent 2 until >= 75% (no cap)
  Agent 4 (UI Engineer)          -> writes index.html
  Agent 5 (Code Reviewer)        -> scores code, loops back to Agent 4 until >= 75% (no cap)
  Agent 6 (Deployment Engine)    -> starts local server + opens browser

A full Markdown log of the agent-to-agent conversation (what each agent
produced, what feedback/critique it received, and its accuracy score) is
saved to .vscode/agent_conversation_log.md after every run.

Requires: pip install groq python-dotenv
API key:  https://console.groq.com/keys (free)
"""

import os
import sys

# Force UTF-8 output so Windows' default console codepage (cp1252/cp437)
# doesn't crash on Unicode characters that models sometimes emit
# (curly quotes, non-breaking spaces, em-dashes, etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")              # optional fallback (free tier)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")  # optional last-resort (Ollama, free/unlimited)

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found.")
    print("1. Get a free key at https://console.groq.com/keys")
    print("2. Create a file named '.env' in this folder with:")
    print('   GROQ_API_KEY=your_key_here')
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)

gemini_configured = False
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_configured = True

# ---------------------------------------------------------------------------
# CROSS-PROVIDER FALLBACK CHAINS: Groq (multiple models) -> Gemini (free) -> Ollama (free, local, unlimited)
# Each agent "role" has an ordered chain. If one entry is rate-limited or
# fails, the NEXT entry is tried immediately - including switching providers.
# Providers with no key configured (Gemini) are skipped automatically.
# ---------------------------------------------------------------------------
FALLBACK_CHAINS = {
    "planner": [
        {"provider": "groq", "model": "openai/gpt-oss-120b"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "groq", "model": "llama-3.1-8b-instant"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "local", "model": "llama3"},
    ],
    "architect": [
        {"provider": "groq", "model": "qwen/qwen3.6-27b"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "groq", "model": "llama-3.1-8b-instant"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "local", "model": "llama3"},
    ],
    "auditor": [
        {"provider": "groq", "model": "qwen/qwen3.6-27b"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "local", "model": "llama3"},
    ],
    "coder": [
        {"provider": "groq", "model": "openai/gpt-oss-120b"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "groq", "model": "llama-3.1-8b-instant"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "local", "model": "llama3"},
    ],
    "reviewer": [
        {"provider": "groq", "model": "openai/gpt-oss-120b"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "local", "model": "llama3"},
    ],
}

ACCURACY_THRESHOLD = 75
MAX_LOOPS = None  # no cap: Phase 1/2 loops retry indefinitely until ACCURACY_THRESHOLD is met

OUTPUT_DIR = Path(".vscode")
SRC_DIR = Path(".vscode")
OUTPUT_DIR.mkdir(exist_ok=True)
SRC_DIR.mkdir(exist_ok=True)

USER_GOAL = "Create a 'Hello Rushikesh' UI with an advanced, modern, animated look using HTML/CSS/JS."


# ---------------------------------------------------------------------------
# CONVERSATION LOG - records every agent-to-agent exchange for this run
# ---------------------------------------------------------------------------

run_log = []  # list of dict entries, written to Markdown at the end


def log_step(agent_name: str, description: str, content: str,
             score: int = None, feedback_sent: str = None):
    """Record one step of the pipeline (an agent producing output, and/or
    handing feedback to another agent)."""
    run_log.append({
        "agent": agent_name,
        "description": description,
        "content": content,
        "score": score,
        "feedback_sent": feedback_sent,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def save_conversation_log(run_history: dict):
    """Write the full agent conversation (with scores + feedback trail) to
    a single readable Markdown file."""
    lines = []
    lines.append("# Agent Conversation Log")
    lines.append("")
    lines.append(f"**Goal:** {USER_GOAL}")
    lines.append(f"**Run started:** {run_log[0]['timestamp'] if run_log else 'N/A'}")
    lines.append(f"**Accuracy threshold:** {ACCURACY_THRESHOLD}%  |  **Loop cap:** none (retries until threshold is met)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, entry in enumerate(run_log, 1):
        lines.append(f"## Step {i} - {entry['agent']}")
        lines.append(f"*{entry['timestamp']}*")
        lines.append("")
        lines.append(f"**What happened:** {entry['description']}")
        lines.append("")
        if entry["score"] is not None:
            verdict = "PASSED" if entry["score"] >= ACCURACY_THRESHOLD else "BELOW THRESHOLD"
            lines.append(f"**Accuracy score:** {entry['score']}%  ({verdict})")
            lines.append("")
        if entry["feedback_sent"]:
            lines.append("**Feedback/suggestions sent to the next agent:**")
            lines.append("> " + entry["feedback_sent"].replace("\n", "\n> "))
            lines.append("")
        lines.append("<details>")
        lines.append("<summary>Full output produced at this step</summary>")
        lines.append("")
        lines.append("```")
        lines.append(entry["content"])
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Accuracy summary table
    lines.append("## Accuracy Summary")
    lines.append("")
    lines.append("| Phase | Loop # | Score | Result |")
    lines.append("|---|---|---|---|")
    for idx, s in enumerate(run_history["phase1"], 1):
        result = "Approved" if s >= ACCURACY_THRESHOLD else "Sent back to Agent 2"
        lines.append(f"| Phase 1: Spec (Agent 2 <-> Agent 3) | {idx} | {s}% | {result} |")
    for idx, s in enumerate(run_history["phase2"], 1):
        result = "Approved" if s >= ACCURACY_THRESHOLD else "Sent back to Agent 4"
        lines.append(f"| Phase 2: Code (Agent 4 <-> Agent 5) | {idx} | {s}% | {result} |")
    lines.append("")

    path = OUTPUT_DIR / "agent_conversation_log.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Log] Full agent conversation saved to {path}")


# ---------------------------------------------------------------------------
# HELPER: call the LLM
# ---------------------------------------------------------------------------

import re

# Known Groq account-level TPM ceilings per model, used only to pre-shrink
# Groq requests before sending (Gemini/Ollama don't need this).
MODEL_TPM_LIMITS = {
    "qwen/qwen3.6-27b": 8000,
    "openai/gpt-oss-120b": 8000,
    "llama-3.3-70b-versatile": 6000,
    "llama-3.1-8b-instant": 6000,
}


def _parse_retry_wait_seconds(error_text: str, default: float = 15.0) -> float:
    """Extract a wait time like 'try again in 8m17.232s' from a rate-limit
    error message. Falls back to `default` (plus buffer) if not found."""
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", error_text)
    if not match:
        return default
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 2


def _call_groq(model, system_prompt, user_prompt, max_tokens, json_mode):
    approx_prompt_tokens = (len(system_prompt) + len(user_prompt)) // 4
    tpm_limit = MODEL_TPM_LIMITS.get(model, 6000)
    safe_max_tokens = max(512, min(max_tokens, tpm_limit - approx_prompt_tokens - 300))

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": safe_max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        kwargs["include_reasoning"] = False
    elif "qwen3.6" in model:
        kwargs["reasoning_effort"] = "none"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_gemini(model, system_prompt, user_prompt, max_tokens, json_mode):
    if not gemini_configured:
        raise RuntimeError("GEMINI_API_KEY not set; Gemini fallback unavailable.")
    generation_config = {"temperature": 0.4, "max_output_tokens": max_tokens}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    gm = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
    response = gm.generate_content(user_prompt, generation_config=generation_config)
    return response.text


def _call_local(model, system_prompt, user_prompt, max_tokens, json_mode):
    """Last-resort fallback: a local Ollama server. No tokens, no rate
    limits, no internet dependency. Requires Ollama running with `model` pulled."""
    import urllib.request
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.4},
    }
    req = urllib.request.Request(
        f"{LOCAL_LLM_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


PROVIDER_CALLERS = {
    "groq": _call_groq,
    "gemini": _call_gemini,
    "local": _call_local,
}


def call_llm(role: str, system_prompt: str, user_prompt: str, json_mode: bool = False,
             max_tokens: int = 4096) -> str:
    """Walks this role's Groq -> Gemini -> Ollama fallback chain. Skips Gemini
    if no key is configured. On rate limits/errors, tries the next entry
    immediately; only sleeps once the whole chain has failed in one pass."""
    chain = FALLBACK_CHAINS[role]
    idx = 0
    attempt = 0
    current_max_tokens = max_tokens

    while True:
        attempt += 1
        entry = chain[idx]
        provider, model = entry["provider"], entry["model"]

        if provider == "gemini" and not gemini_configured:
            idx = (idx + 1) % len(chain)
            continue

        try:
            result = PROVIDER_CALLERS[provider](model, system_prompt, user_prompt,
                                                 current_max_tokens, json_mode)
            if idx != 0:
                print(f"  [info] Succeeded using fallback: {provider}/{model}")
            return result
        except Exception as e:
            print(f"  [warning] {provider}/{model} failed (attempt {attempt}): {e}")
            err_str = str(e)
            is_rate_limit = ("rate_limit_exceeded" in err_str or "tokens per" in err_str
                              or "429" in err_str or "quota" in err_str.lower())

            if idx + 1 < len(chain):
                idx += 1
                current_max_tokens = max_tokens
                print(f"  [info] Switching to next fallback: {chain[idx]['provider']}/{chain[idx]['model']}")
                continue

            # Whole chain exhausted this pass.
            if is_rate_limit:
                wait_seconds = _parse_retry_wait_seconds(err_str)
                print(f"  [info] All providers exhausted. Waiting {wait_seconds:.0f}s, then retrying from the top...")
                time.sleep(wait_seconds)
            else:
                backoff = min(30, 2 * attempt)
                print(f"  [info] All providers failed. Retrying from the top in {backoff}s...")
                time.sleep(backoff)
                if json_mode:
                    user_prompt += ("\n\nIMPORTANT: Respond with ONLY a single valid JSON object. "
                                     "No markdown, no code fences, no extra text before or after.")
            idx = 0
            current_max_tokens = max_tokens


def extract_code_block(text: str) -> str:
    """Pull raw HTML out of a ```html ... ``` fence if present. Falls back to
    locating <!DOCTYPE or <html directly, in case a reasoning model added
    commentary before/after the fence (or skipped the fence entirely)."""
    if "```html" in text:
        candidate = text.split("```html")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        candidate = parts[1].strip() if len(parts) >= 2 else text.strip()
    else:
        candidate = text.strip()

    # Safety net: if what we extracted doesn't look like HTML, search the
    # full text for the first real HTML tag and cut from there.
    lower = candidate.lower()
    if "<!doctype" not in lower and "<html" not in lower:
        lower_full = text.lower()
        idx = lower_full.find("<!doctype")
        if idx == -1:
            idx = lower_full.find("<html")
        if idx != -1:
            candidate = text[idx:].strip()
            if "```" in candidate:
                candidate = candidate.split("```")[0].strip()

    return candidate


# ---------------------------------------------------------------------------
# AGENT 1: Requirement Planner
# ---------------------------------------------------------------------------

def agent1_requirement_planner(goal: str) -> str:
    print("\n[Agent 1] Requirement Planner is working...")
    system = (
        "You are a requirements analyst. Turn a one-line user goal into a clear, "
        "structured requirements document in Markdown. Include: Overview, "
        "Functional Requirements, Visual/UX Requirements, Technical Constraints."
    )
    result = call_llm("planner", system, f"User goal: {goal}")
    path = OUTPUT_DIR / "requirements.md"
    path.write_text(result, encoding="utf-8")
    print(f"[Agent 1] Done. Requirements saved to {path}")
    print("-" * 60)
    print(result)
    print("-" * 60)

    log_step(
        agent_name="Agent 1 - Requirement Planner",
        description=f"Turned the raw user goal into a structured requirements.md.",
        content=result,
    )
    return result


# ---------------------------------------------------------------------------
# AGENT 2: System Architect
# ---------------------------------------------------------------------------

def agent2_system_architect(requirements: str, critique: str = None) -> str:
    print("\n[Agent 2] System Architect is working...")
    system = (
        "You are a senior frontend architect. Turn requirements into a DETAILED "
        "technical specification in Markdown for a single-page HTML/CSS/JS app. "
        "Include: Layout structure, Color palette, Typography, Animations/transitions, "
        "Component breakdown, Responsive behavior. Be specific and advanced.\n\n"
        "CRITICAL CONTRAST RULE: When specifying the color palette, the text color "
        "must have strong contrast against the background at EVERY point it appears "
        "(minimum WCAG AA contrast ratio of 4.5:1). If the background is a gradient, "
        "pick a text color (typically white or near-black) that stays readable across "
        "the ENTIRE gradient range, not just one end of it. Do not pick a text color "
        "that is close in hue or lightness to any part of the background. State the "
        "exact hex codes for both and briefly justify why they contrast well."
    )
    user = f"Requirements:\n{requirements}"
    if critique:
        user += f"\n\nThe previous version of this spec was reviewed and needs improvement:\n{critique}\nPlease produce an improved, more detailed version."

    result = call_llm("architect", system, user, max_tokens=8192)
    if not result.strip():
        print("[Agent 2] FATAL: No spec was generated after retries. Stopping.")
        sys.exit(1)
    path = OUTPUT_DIR / "document.md"
    path.write_text(result, encoding="utf-8")
    print(f"[Agent 2] Done. Spec saved to {path}")
    print("-" * 60)
    print(result)
    print("-" * 60)

    description = (
        "Read Agent 1's requirements.md and wrote the technical spec as document.md."
        if not critique else
        "Revised document.md based on Agent 3's critique (see feedback Agent 2 received, below)."
    )
    log_step(
        agent_name="Agent 2 - System Architect",
        description=description,
        content=result,
        feedback_sent=None,  # feedback Agent 2 *received* is logged on Agent 3's entry
    )
    return result


# ---------------------------------------------------------------------------
# AGENT 3: QA Auditor (checklist-based scoring, not a guessed %)
# ---------------------------------------------------------------------------

def agent3_qa_auditor(requirements: str, spec: str) -> dict:
    print("\n[Agent 3] QA Auditor is checking the spec...")
    system = (
        "You are a meticulous QA auditor reviewing a technical spec against requirements. "
        "Score EACH checklist item as true/false, then return ONLY valid JSON in this exact shape:\n"
        '{"checklist": {"covers_all_functional_requirements": true/false, '
        '"defines_layout_clearly": true/false, "defines_visual_style": true/false, '
        '"defines_animations": true/false, "defines_responsiveness": true/false, '
        '"technically_feasible": true/false, "document_is_complete": true/false}, '
        '"critique": "short text explaining any false items"}\n\n'
        "For document_is_complete: check whether the spec ends properly (e.g. closes its "
        "last section/sentence/code block) rather than stopping abruptly mid-word, "
        "mid-sentence, or mid-list. If it looks cut off, mark this false."
    )
    user = f"Requirements:\n{requirements}\n\nSpec to review:\n{spec}"
    raw = call_llm("auditor", system, user, json_mode=True)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"checklist": {}, "critique": "Could not parse auditor output; defaulting to fail."}

    checklist = data.get("checklist", {})
    total = len(checklist) if checklist else 1
    passed = sum(1 for v in checklist.values() if v is True)
    score = round((passed / total) * 100)

    data["score"] = score
    print(f"[Agent 3] Score: {score}%  (passed {passed}/{total} checklist items)")
    print("-" * 60)
    for item, ok in checklist.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {item}")
    print(f"  Critique: {data.get('critique', '(none)')}")
    print("-" * 60)

    checklist_report = "\n".join(
        f"- [{'x' if ok else ' '}] {item}" for item, ok in checklist.items()
    )
    full_content = f"Checklist:\n{checklist_report}\n\nCritique: {data.get('critique', '(none)')}"

    # Save Agent 3's own report to disk, same as every other agent's output.
    verdict = "PASSED" if score >= ACCURACY_THRESHOLD else "BELOW THRESHOLD"
    report_md = (
        f"# QA Audit Report\n\n"
        f"**Score:** {score}%  ({verdict})\n\n"
        f"## Checklist\n{checklist_report}\n\n"
        f"## Critique\n{data.get('critique', '(none)')}\n"
    )
    report_path = OUTPUT_DIR / "audit_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[Agent 3] Report saved to {report_path}")

    log_step(
        agent_name="Agent 3 - QA Auditor",
        description="Scored Agent 2's document.md against the checklist.",
        content=full_content,
        score=score,
        feedback_sent=data.get("critique") if score < ACCURACY_THRESHOLD else None,
    )
    return data


# ---------------------------------------------------------------------------
# AGENT 4: Lead UI Engineer
# ---------------------------------------------------------------------------

def agent4_ui_engineer(spec: str, feedback: str = None) -> str:
    print("\n[Agent 4] UI Engineer is generating code...")
    system = (
        "You are an expert frontend engineer. Generate a SINGLE self-contained HTML file "
        "(inline CSS in <style>, inline JS in <script>) implementing the given spec. "
        "Make it visually advanced: gradients, smooth animations, modern layout. "
        "Return ONLY the code in a ```html fenced block, nothing else."
    )
    user = f"Spec:\n{spec}"
    if feedback:
        user += f"\n\nThe previous version of this code was reviewed and needs improvement:\n{feedback}\nPlease produce an improved, more advanced version of the FULL file."

    raw = call_llm("coder", system, user, max_tokens=8192)
    if not raw.strip():
        print("[Agent 4] FATAL: No code was generated after retries. Stopping.")
        sys.exit(1)
    code = extract_code_block(raw)
    path = SRC_DIR / "index.html"
    path.write_text(code, encoding="utf-8")
    print(f"[Agent 4] Done. Code saved to {path}")
    print("-" * 60)
    print(code)
    print("-" * 60)

    description = (
        "Generated index.html from Agent 2's document.md."
        if not feedback else
        "Revised index.html based on Agent 5's feedback (see feedback Agent 4 received, below)."
    )
    log_step(
        agent_name="Agent 4 - UI Engineer",
        description=description,
        content=code,
    )
    return code


# ---------------------------------------------------------------------------
# AGENT 5: Senior Code Reviewer (checklist-based scoring)
# ---------------------------------------------------------------------------

def agent5_code_reviewer(spec: str, code: str) -> dict:
    print("\n[Agent 5] Code Reviewer is checking the code...")
    system = (
        "You are a senior code reviewer. Score EACH checklist item as true/false for the "
        "given HTML/CSS/JS code, then return ONLY valid JSON in this exact shape:\n"
        '{"checklist": {"valid_html_structure": true/false, "matches_spec_layout": true/false, '
        '"has_animations_or_transitions": true/false, "responsive_design": true/false, '
        '"no_obvious_bugs": true/false, "clean_code_structure": true/false, '
        '"text_has_strong_contrast": true/false}, '
        '"feedback": "short actionable text on what to improve if anything failed"}\n\n'
        "For text_has_strong_contrast: check the actual hex/rgb color values in the CSS. "
        "Mentally verify the text color has at least WCAG AA contrast (4.5:1) against the "
        "background at every point, including all stops of any gradient. If text color is "
        "close in hue or lightness to the background anywhere, mark this false and explain "
        "exactly which colors clash in the feedback."
    )
    user = f"Spec:\n{spec}\n\nCode to review:\n{code}"
    raw = call_llm("reviewer", system, user, json_mode=True)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"checklist": {}, "feedback": "Could not parse reviewer output; defaulting to fail."}

    checklist = data.get("checklist", {})
    total = len(checklist) if checklist else 1
    passed = sum(1 for v in checklist.values() if v is True)
    score = round((passed / total) * 100)

    data["score"] = score
    print(f"[Agent 5] Score: {score}%  (passed {passed}/{total} checklist items)")
    print("-" * 60)
    for item, ok in checklist.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {item}")
    print(f"  Feedback: {data.get('feedback', '(none)')}")
    print("-" * 60)

    checklist_report = "\n".join(
        f"- [{'x' if ok else ' '}] {item}" for item, ok in checklist.items()
    )
    full_content = f"Checklist:\n{checklist_report}\n\nFeedback: {data.get('feedback', '(none)')}"

    # Save Agent 5's own report to disk, same as every other agent's output.
    verdict = "PASSED" if score >= ACCURACY_THRESHOLD else "BELOW THRESHOLD"
    report_md = (
        f"# Code Review Report\n\n"
        f"**Score:** {score}%  ({verdict})\n\n"
        f"## Checklist\n{checklist_report}\n\n"
        f"## Feedback\n{data.get('feedback', '(none)')}\n"
    )
    report_path = OUTPUT_DIR / "review_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[Agent 5] Report saved to {report_path}")

    log_step(
        agent_name="Agent 5 - Code Reviewer",
        description="Scored Agent 4's code against the checklist.",
        content=full_content,
        score=score,
        feedback_sent=data.get("feedback") if score < ACCURACY_THRESHOLD else None,
    )
    return data


# ---------------------------------------------------------------------------
# AGENT 6: Deployment Engine (no LLM - just runs a local server)
# ---------------------------------------------------------------------------

def agent6_deployment_engine(port: int = 8000):
    import http.server
    import socketserver
    import threading
    import webbrowser

    print("\n[Agent 6] Deployment Engine starting local server...")
    os.chdir(SRC_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    def serve():
        httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/index.html"
    print(f"[Agent 6] Server running at {url}")
    webbrowser.open(url)
    print("[Agent 6] Browser opened. Press Ctrl+C to stop the server.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Agent 6] Shutting down server.")
        httpd.shutdown()


# ---------------------------------------------------------------------------
# ORCHESTRATOR - runs the whole pipeline automatically
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AGENTIC SDLC PIPELINE STARTING")
    print("=" * 60)

    # Agent 1
    requirements = agent1_requirement_planner(USER_GOAL)

    run_history = {"phase1": [], "phase2": []}

    # Phase 1: Agent 2 <-> Agent 3 loop
    # Keeps retrying with NO cap until the spec actually reaches ACCURACY_THRESHOLD.
    # A below-threshold spec is never accepted or passed along to Phase 2.
    critique = None
    spec = None
    i = 0
    while True:
        i += 1
        print(f"\n--- Phase 1 Loop {i} ---")
        spec = agent2_system_architect(requirements, critique)
        audit = agent3_qa_auditor(requirements, spec)
        run_history["phase1"].append(audit["score"])
        if audit["score"] >= ACCURACY_THRESHOLD:
            print(f"[Phase 1] Spec approved at {audit['score']}% after {i} loop(s). Moving to Phase 2.")
            break
        critique = audit.get("critique", "Please improve the spec.")
        print(f"[Phase 1] Spec below threshold ({audit['score']}%). Sending back to Agent 2 (attempt {i + 1} next).")

    # Phase 2: Agent 4 <-> Agent 5 loop
    # Same rule: no cap, no fallback. Code only moves to deployment once it
    # actually clears ACCURACY_THRESHOLD.
    feedback = None
    code = None
    j = 0
    while True:
        j += 1
        print(f"\n--- Phase 2 Loop {j} ---")
        code = agent4_ui_engineer(spec, feedback)
        review = agent5_code_reviewer(spec, code)
        run_history["phase2"].append(review["score"])
        if review["score"] >= ACCURACY_THRESHOLD:
            print(f"[Phase 2] Code approved at {review['score']}% after {j} loop(s). Moving to deployment.")
            break
        feedback = review.get("feedback", "Please improve the code.")
        print(f"[Phase 2] Code below threshold ({review['score']}%). Sending back to Agent 4 (attempt {j + 1} next).")

    # Save the full agent-to-agent conversation + accuracy summary as Markdown
    save_conversation_log(run_history)

    # Agent 6
    agent6_deployment_engine()


if __name__ == "__main__":
    main()