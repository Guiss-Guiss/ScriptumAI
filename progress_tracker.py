# progress_tracker.py
import streamlit as st
import threading

class ProgressTracker:
    def __init__(self, total_files):
        self.total_files = total_files
        self.processed_files = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.lock = threading.Lock()

    def update(self, file_name):
        with self.lock:
            self.processed_files += 1
            progress = self.processed_files / self.total_files
            self.progress_bar.progress(progress)
            self.status_text.text(f"Processing: {file_name} ({self.processed_files}/{self.total_files})")

    def complete(self):
        self.progress_bar.progress(1.0)
        self.status_text.text("All files processed successfully!")