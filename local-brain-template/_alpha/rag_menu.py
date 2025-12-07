#!/usr/bin/env python3
"""
rag_menu.py

Interactive CLI menu for your local LLM + RAG toolkit.

Provides:
- Quick access to common scripts (unfinished projects, taxonomy, profile, etc.)
- A 'help screen' showing all relevant commands and what they do
"""

import sys
import subprocess
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_subprocess(cmd, cwd=None):
    """Run a subprocess and stream output."""
    try:
        subprocess.run(cmd, cwd=cwd or HERE, check=False)
    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user.")


def clear_screen():
    # Cross-platform-ish clear
    subprocess.run(["clear"], check=False)


def print_header():
    print("=" * 70)
    print("  Local LLM + RAG Control Panel".center(70))
    print("=" * 70)
    print()
    print("Workspace:", HERE)
    print()


def print_help_reference():
    """Show all currently available commands / scripts with descriptions."""
    print_header()
    print("Currently available commands and scripts:\n")

    cmds = [
        (
            "llm \"<prompt>\"",
            "Run a one-off RAG-augmented query using lm_studio + Chroma."
        ),
        (
            "python rag_index.py",
            "Rebuild the entire RAG index from your ChatGPT export."
        ),
        (
            "python rag_watch.py",
            "Run a filesystem watcher that rebuilds the index when export changes."
        ),
        (
            "python lm_ping.py",
            "Ping LM Studio's OpenAI-compatible server and verify connectivity."
        ),
        (
            "python unfinished_projects.py",
            "Generate a Markdown report of unfinished projects inferred from your history."
        ),
        (
            "python topic_taxonomy.py",
            "Generate a topic taxonomy report (what you tend to work on / think about)."
        ),
        (
            "python generate_profile.py",
            "Generate or refresh profile.md summarizing your environment & preferences."
        ),
        (
            "systemctl --user status rag-watch.service",
            "Check status of the background RAG watcher (if enabled)."
        ),
        (
            "systemctl --user start|stop|restart rag-watch.service",
            "Manually control the background RAG watcher service."
        ),
        (
            "python ingest_conversations.py",
            "Ingest new CLI conversations from data/conversations.jsonl into the RAG index."
        ),

    ]

    for cmd, desc in cmds:
        print(f"  {cmd}")
        wrapped = textwrap.fill(desc, width=68, subsequent_indent="      ")
        print(f"      {wrapped}\n")

    print("-" * 70)
    print("Use this screen as a quick reference for your LLM/RAG tooling.")
    print("More entries will be added here as we build new scripts.\n")


def menu():
    """Main interactive menu loop."""
    while True:
        clear_screen()
        print_header()

        print("Choose an option:\n")
        print("  1) Run ad-hoc RAG query (llm)")
        print("  2) Generate unfinished projects report")
        print("  3) Generate topic taxonomy report")
        print("  4) Generate or update profile.md")
        print("  5) Ping LM Studio server (health check)")
        print("  6) Rebuild RAG index from ChatGPT export")
        print("  7) Ingest new CLI conversations into RAG")
        print("  8) Show help / reference screen")
        print("  q) Quit")

        choice = input("Select an option: ").strip().lower()

        if choice in ("q", "quit", "x", "exit"):
            print("\nGoodbye.")
            return

        if choice == "1":
            handle_ad_hoc_query()
        elif choice == "2":
            handle_unfinished_projects()
        elif choice == "3":
            handle_topic_taxonomy()
        elif choice == "4":
            handle_profile()
        elif choice == "5":
            handle_ping()
        elif choice == "6":
            handle_rebuild_index()
        elif choice == "7":
            handle_ingest_conversations()
        elif choice == "8":
            clear_screen()
            print_help_reference()
            input("\nPress Enter to return to menu...")
        else:
            print("\n[WARN] Invalid choice. Press Enter to try again.")
            input()
            continue


def handle_ad_hoc_query():
    clear_screen()
    print_header()
    print("Ad-hoc RAG query")
    print("-" * 70)
    print("This will call the llm_rag_cli.py pipeline using your current venv.\n")
    prompt = input("Enter your prompt (empty to cancel): ").strip()
    if not prompt:
        return

    # Use the same Python interpreter that is running this script
    cmd = [sys.executable, str(HERE / "llm_rag_cli.py"), prompt]
    print("\n[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")


def handle_unfinished_projects():
    clear_screen()
    print_header()
    print("Unfinished projects report")
    print("-" * 70)
    print("This runs unfinished_projects.py and prints the Markdown report.\n")
    cmd = [sys.executable, str(HERE / "unfinished_projects.py")]
    print("[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")


def handle_topic_taxonomy():
    clear_screen()
    print_header()
    print("Topic taxonomy report")
    print("-" * 70)
    print("This runs topic_taxonomy.py and prints the Markdown report.\n")
    cmd = [sys.executable, str(HERE / "topic_taxonomy.py")]
    print("[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")


def handle_profile():
    clear_screen()
    print_header()
    print("Generate / update profile.md")
    print("-" * 70)
    print("This runs generate_profile.py to write or refresh profile.md.\n")
    cmd = [sys.executable, str(HERE / "generate_profile.py")]
    print("[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")


def handle_ping():
    clear_screen()
    print_header()
    print("LM Studio connectivity check")
    print("-" * 70)
    print("This runs lm_ping.py to verify the LM Studio API is reachable.\n")
    cmd = [sys.executable, str(HERE / "lm_ping.py")]
    print("[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")


def handle_rebuild_index():
    clear_screen()
    print_header()
    print("Rebuild RAG index")
    print("-" * 70)
    print("This runs rag_index.py to fully rebuild the Chroma index.\n")
    confirm = input("This may take a while. Proceed? [y/N]: ").strip().lower()
    if confirm != "y":
        return
    cmd = [sys.executable, str(HERE / "rag_index.py")]
    print("[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")

def handle_ingest_conversations():
    clear_screen()
    print_header()
    print("Ingest new CLI conversations into RAG")
    print("-" * 70)
    print("This runs ingest_conversations.py to add new logged chats into the index.\n")
    cmd = [sys.executable, str(HERE / "ingest_conversations.py")]
    print("[INFO] Running:", " ".join(cmd), "\n")
    run_subprocess(cmd)
    input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    menu()
