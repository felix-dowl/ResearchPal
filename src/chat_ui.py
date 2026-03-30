from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from conversation_handler import ConversationHandler


class ResearchPalChatUI:
    def __init__(self) -> None:
        self.handler = ConversationHandler()

        self.root = tk.Tk()
        self.root.title("ResearchPal Chat")
        self.root.geometry("900x700")

        intake_frame = tk.LabelFrame(self.root, text="Add Sources")
        intake_frame.pack(fill=tk.X, padx=12, pady=(12, 8))

        url_frame = tk.Frame(intake_frame)
        url_frame.pack(fill=tk.X, padx=10, pady=(10, 8))

        self.url_input = tk.Entry(url_frame)
        self.url_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.url_input.bind("<Return>", self.submit_url)

        self.add_url_button = tk.Button(url_frame, text="Add URL", command=self.submit_url)
        self.add_url_button.pack(side=tk.LEFT, padx=(8, 0))

        file_frame = tk.Frame(intake_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.add_files_button = tk.Button(file_frame, text="Add Files Or Folder", command=self.select_sources)
        self.add_files_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.chat_history = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.query_input = tk.Entry(controls)
        self.query_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.query_input.bind("<Return>", self.submit_query)

        self.send_button = tk.Button(controls, text="Send", command=self.submit_query)
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))

        self.pending_response_start: str | None = None

    def append_message(self, speaker: str, message: str) -> None:
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{speaker}:\n{message}\n\n")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.see(tk.END)

    def normalize_assistant_answer(self, answer_text: str) -> str:
        cleaned = answer_text.strip()

        while cleaned.startswith("ResearchPal:"):
            cleaned = cleaned.removeprefix("ResearchPal:").lstrip()

        return cleaned

    def show_error(self, title: str, message: str) -> None:
        self.append_message("System", f"{title}: {message}")
        messagebox.showerror(title, message)

    def set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.query_input.config(state=state)
        self.send_button.config(state=state)
        self.url_input.config(state=state)
        self.add_url_button.config(state=state)
        self.add_files_button.config(state=state)

    def submit_url(self, event=None) -> None:
        url = self.url_input.get().strip()
        if not url:
            return

        self.url_input.delete(0, tk.END)
        self.set_busy(True)
        self.append_message("System", f"Adding URL: {url}")
        threading.Thread(target=self.run_ingestion, args=([url],), daemon=True).start()

    def select_sources(self) -> None:
        selected_files = list(
            filedialog.askopenfilenames(
                title="Select files to ingest",
                filetypes=[
                    ("Supported files", "*.pdf *.txt *.html *.htm"),
                    ("All files", "*.*"),
                ],
            )
        )
        selected_folder = filedialog.askdirectory(title="Or select a folder to ingest")

        sources = selected_files.copy()
        if selected_folder:
            sources.append(selected_folder)

        if not sources:
            return

        self.set_busy(True)
        self.append_message("System", f"Adding {len(sources)} source(s).")
        threading.Thread(target=self.run_ingestion, args=(sources,), daemon=True).start()

    def run_ingestion(self, sources: list[str]) -> None:
        try:
            results = []
            for source in sources:
                results.append(self.handler.ingest(source))
            self.root.after(0, self.finish_ingestion, sources, results)
        except Exception as exc:
            self.root.after(0, self.finish_ingestion_error, sources, exc)

    def finish_ingestion(self, sources: list[str], results: list) -> None:
        messages = []
        for source, result in zip(sources, results):
            if hasattr(result, "stored_count"):
                messages.append(f"{source} -> stored {result.stored_count} chunks.")
            else:
                messages.append(f"{source} -> ingestion completed.")

        self.append_message("System", "\n".join(messages))
        self.set_busy(False)

    def finish_ingestion_error(self, sources: list[str], error: Exception) -> None:
        self.set_busy(False)
        joined_sources = ", ".join(sources)
        self.show_error("Ingestion failed", f"Could not ingest: {joined_sources}\n{error}")

    def submit_query(self, event=None) -> None:
        question = self.query_input.get().strip()
        if not question:
            return

        self.query_input.delete(0, tk.END)
        self.append_message("User", question)
        self.set_busy(True)
        self.chat_history.config(state=tk.NORMAL)
        self.pending_response_start = self.chat_history.index(tk.END)
        self.chat_history.insert(tk.END, "ResearchPal:\nThinking...\n\n")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.see(tk.END)

        threading.Thread(target=self.run_query, args=(question,), daemon=True).start()

    def run_query(self, question: str) -> None:
        try:
            response = self.handler.ask(question, method="mmr", k=5, rewrite_query=False)
            self.root.after(0, self.finish_query, response.answer)
        except Exception as exc:
            self.root.after(0, self.finish_query_error, exc)

    def finish_query(self, answer_text: str) -> None:
        if self.pending_response_start is not None:
            self.chat_history.config(state=tk.NORMAL)
            self.chat_history.delete(self.pending_response_start, tk.END)
            self.chat_history.config(state=tk.DISABLED)
            self.pending_response_start = None

        self.append_message("ResearchPal", self.normalize_assistant_answer(answer_text))
        self.set_busy(False)

    def finish_query_error(self, error: Exception) -> None:
        if self.pending_response_start is not None:
            self.chat_history.config(state=tk.NORMAL)
            self.chat_history.delete(self.pending_response_start, tk.END)
            self.chat_history.config(state=tk.DISABLED)
            self.pending_response_start = None

        self.set_busy(False)
        self.show_error("Query failed", str(error))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = ResearchPalChatUI()
    app.run()
