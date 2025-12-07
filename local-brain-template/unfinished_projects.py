#!/usr/bin/env python3
"""
unfinished_projects.py

Use the existing RAG + LM Studio pipeline to infer unfinished projects
from your historical conversations, and output a Markdown report.
"""

from copy import deepcopy
import textwrap

from llm_rag_cli import (
    load_config,
    get_collection,
    retrieve_context,
    call_lm_studio,
)


def main():
    cfg = load_config()

    # Make a shallow copy so we can bump context size just for this report
    cfg_big = deepcopy(cfg)
    cfg_big.setdefault("rag", {})
    cfg_big["rag"]["max_context_chars"] = int(
        cfg_big["rag"].get("max_context_chars", 8000) * 2
    )

    collection = get_collection(cfg_big)

    # Query designed to pull out todo/unfinished-ish bits
    query = (
        "tasks, plans, projects, things to do later, TODOs, unfinished work, "
        "projects I said we'd come back to or finish later"
    )

    context = retrieve_context(collection, query, cfg_big)

    if not context:
        print("# Unfinished Projects Report\n")
        print("_No relevant context was retrieved from the RAG index._")
        return

    system_prompt = "\n".join(
        [
            "You are an assistant that reviews my past technical conversations and plans.",
            "You will be given context snippets from my own chat history.",
            "Your job is to infer which projects or initiatives appear to be started but not clearly finished.",
            "Use only the provided context; do not invent projects that aren't supported by it.",
            "Be concise but concrete. Focus on next actions I can actually take.",
        ]
    )

    user_message = textwrap.dedent(
        f"""\
        You are given context from my historical conversations:

        {context}

        Based ONLY on this context:

        1. Identify projects, tasks, or initiatives that appear to be unfinished or in-progress.
        2. Group them by project (e.g., "Proxmox cluster reconfiguration", "OPNsense VLAN design").
        3. For each project, list:
           - A short description (1–2 sentences)
           - Why you think it is unfinished (evidence from the context)
           - 3–5 concrete next actions I can take to move it forward.
        4. If you are unsure whether something is finished, err on the side of including it but mark it as **uncertain**.
        5. Output in Markdown with clear headings, like:

           # Unfinished Projects
           ## Project: ...
           - Status: ...
           - Evidence: ...
           - Next actions:
             - ...
             - ...

        Do not mention the internal mechanics of RAG or embeddings. Talk as if you just remember my history.
        """
    )

    report = call_lm_studio(cfg_big, user_message, system_prompt=system_prompt)
    print(report)


if __name__ == "__main__":
    main()
