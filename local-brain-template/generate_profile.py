#!/usr/bin/env python3
"""
generate_profile.py

Generate a profile.md summarizing who I am, what I work on, and my preferences,
based on my RAG index + LM Studio.
"""

from copy import deepcopy
import pathlib
import textwrap

from llm_rag_cli import (
    load_config,
    get_collection,
    retrieve_context,
    call_lm_studio,
)


HERE = pathlib.Path(__file__).resolve().parent
PROFILE_PATH = HERE / "profile.md"


def main():
    cfg = load_config()
    cfg_big = deepcopy(cfg)
    cfg_big.setdefault("rag", {})
    cfg_big["rag"]["max_context_chars"] = int(
        cfg_big["rag"].get("max_context_chars", 8000) * 2
    )

    collection = get_collection(cfg_big)

    # Query tuned to grab 'who I am' type content
    query = (
        "what I work on, my homelab, my operating systems, my tools, my preferences, "
        "my goals, recurring projects, frustrations, plans, and questions"
    )

    context = retrieve_context(collection, query, cfg_big)

    if not context:
        print("[WARN] No context retrieved; profile will be very generic.")
        context = "(No RAG context available.)"

    system_prompt = "\n".join(
        [
            "You are creating a personal technical profile for me.",
            "This profile will be used as a reference and possibly as a system prompt for local LLMs.",
            "Be accurate, cautious about over-claiming, and explicit about uncertainties.",
        ]
    )

    user_message = textwrap.dedent(
        f"""\
        You are given context snippets from my historical chats:

        {context}

        Based ONLY on this context (and general reasoning), write a Markdown file called profile.md
        that summarizes who I am and how I work, structured roughly as follows:

        # Profile

        ## Overview
        - Short, 2–4 sentence description of who I seem to be, in practical terms.

        ## Technical Environment
        - Operating systems I use (with any notable distributions or configurations).
        - Homelab / infrastructure components (hypervisors, firewalls, NAS, etc.).
        - Key tools and technologies (editors, shells, terminals, stacks).

        ## Skills & Experience (Inferred)
        - A bullet list of areas where I appear to have hands-on experience.
        - Phrase these as 'appears to have experience with X' rather than absolute claims.

        ## Preferences & Work Style
        - Things I seem to prefer in tools (e.g. CLI-centric, automation, offline, etc.).
        - How I tend to approach troubleshooting or projects.

        ## Current & Ongoing Focus Areas
        - Projects or themes that appear active or recurrent (e.g., optimizing Linux desktop, homelab networking).
        - For each, a sentence about why it matters for me.

        ## How to Help Me (for Future Assistants)
        - Concrete guidelines for how an AI assistant can be maximally useful to me.
        - Include preferences like level of detail, step-by-step vs high-level, etc.

        Make sure this file:
        - Does NOT include any private secrets or identifiers (no passwords, tokens, etc.).
        - Does NOT speculate about real-world identity beyond what is implied by the text.
        - Is written in a neutral, practical tone.
        - Is self-contained and useful as a reference.

        Output ONLY the Markdown content for profile.md.
        """
    )

    content = call_lm_studio(cfg_big, user_message, system_prompt=system_prompt)

    PROFILE_PATH.write_text(content, encoding="utf-8")
    print(f"[INFO] Wrote profile to {PROFILE_PATH}")


if __name__ == "__main__":
    main()
