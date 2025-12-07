#!/usr/bin/env python3
"""
topic_taxonomy.py

Generate a topic taxonomy (buckets) from your RAG index via LM Studio.
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
    cfg_big = deepcopy(cfg)
    cfg_big.setdefault("rag", {})
    cfg_big["rag"]["max_context_chars"] = int(
        cfg_big["rag"].get("max_context_chars", 8000) * 2
    )

    collection = get_collection(cfg_big)

    # Broad query aimed at retrieving a mix of themes
    query = (
        "overall summary of my projects, topics, conversations, technical areas, "
        "recurring themes, homelab, networking, linux, optimization, family, planning"
    )

    context = retrieve_context(collection, query, cfg_big)

    if not context:
        print("# Topic Taxonomy\n")
        print("_No context retrieved; index may be empty or query too narrow._")
        return

    system_prompt = "\n".join(
        [
            "You are an expert at information architecture and topic modeling.",
            "You will analyze snippets from my historical chats and derive a topic taxonomy.",
            "The goal is to give me a clear map of what I work on and think about.",
        ]
    )

    user_message = textwrap.dedent(
        f"""\
        You are given context snippets from my personal chat history:

        {context}

        Based ONLY on these snippets (and general reasoning):

        1. Propose a topic taxonomy: 8–15 top-level topics that cover most of the content.
           Examples of possible topics include: 'Homelab & Proxmox', 'Networking & Firewalls',
           'Linux desktop optimization', 'Storage & NAS', 'Family life & logistics', etc.
        2. For each topic, provide:
           - A short description (what this topic is about for me specifically)
           - 3–7 representative example subtopics or project types
        3. Then, add a section 'How to Use This Taxonomy' with practical ideas for:
           - Organizing notes or an Obsidian vault
           - Tagging future conversations or documents
           - Prioritizing which topics matter most for my goals
        4. Present the output as Markdown, e.g.:

           # Topic Taxonomy
           ## 1. Topic Name
           - Description: ...
           - Representative subtopics:
             - ...
             - ...

           ## How to Use This Taxonomy
           - ...

        Focus on clarity and usefulness; avoid buzzwords.
        """
    )

    report = call_lm_studio(cfg_big, user_message, system_prompt=system_prompt)
    print(report)


if __name__ == "__main__":
    main()
