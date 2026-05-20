"""
agentClaude.py
GUI chatbot interface using the Anthropic Claude API.
Configuration is loaded from config.json (CONFIG section: user, name, instructions,
optional claude_model). Drag-and-drop a PDF or image onto the window to attach it
to the conversation.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import os
import sys
import io
import json
import glob
import argparse
import base64
import mimetypes
from datetime import datetime
import anthropic
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QVBoxLayout, QPushButton,
    QHBoxLayout, QLabel, QComboBox, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont
from md_widget import MarkdownTextEdit
from md_loader import load_persona
from file_upload_worker import FileUploadWorker, format_usage


# Anthropic deprecated temperature on Opus 4.7+. Returns {'temperature': v}
# only for models that still accept it.
_TEMPERATURE_DEPRECATED_PREFIXES = ("claude-opus-4-7",)


def _temperature_kwarg(model, value):
    for prefix in _TEMPERATURE_DEPRECATED_PREFIXES:
        if model.startswith(prefix):
            return {}
    return {"temperature": value}

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class LLMWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, chatbot):
        super().__init__()
        self.user_input = user_input
        self.chatbot = chatbot  # owns client, model, instructions, history

    def run(self):
        try:
            self.chatbot.history.append({"role": "user", "content": self.user_input})
            response = self.chatbot.client.messages.create(
                model=self.chatbot.model,
                max_tokens=self.chatbot.max_tokens,
                system=self.chatbot.instructions,
                messages=self.chatbot.history,
                **_temperature_kwarg(self.chatbot.model, self.chatbot.temperature),
            )
            content = response.content[0].text
            self.chatbot.history.append({"role": "assistant", "content": content})
            self.result_ready.emit(content)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")


class ClaudeChatbot(QWidget):
    def __init__(self, role_md_path="general.md"):
        super().__init__()

        # Load configuration from CONFIG section
        config_file = "config.json"
        with open(config_file, 'r', encoding='utf-8') as file:
            raw_config = json.load(file)
            config = raw_config['CONFIG']

        # Extract configuration values
        self.user = config['user']
        self.name = config['name']

        # Build instructions from common.md + role markdown with variable substitution
        self.common_md = config.get('common_md', 'common.md')
        self.role_md = role_md_path
        self.instructions = load_persona(self.common_md, self.role_md, {
            "user": self.user,
            "name": self.name,
        })
        self.model = config.get('claude_model', 'claude-sonnet-4-5-20250929')
        self.max_tokens = int(config.get('max_tokens', 1000))
        self.temperature = float(config.get('temperature', 0.7))
        self.font_size = int(config.get('fontsize', 12))
        self.loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.latest_response = ""

        # Load API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
            exit(1)

        # Initialize Anthropic client and conversation history
        self.client = anthropic.Anthropic(api_key=api_key)
        self.history = []

        # Set up the GUI interface
        self.init_gui()

    def init_gui(self):
        # Set up main window parameters
        self.setWindowTitle("JuanClaude")
        self.setGeometry(100, 100, 700, 500)
        self.setAcceptDrops(True)  # Enable drag and drop

        layout = QVBoxLayout()

        # --- Header row: Role dropdown (left) + Font +/- (right) ---
        header = QHBoxLayout()
        header.addWidget(QLabel("Role:"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(self._discover_md_files())
        if self.role_md in [self.role_combo.itemText(i) for i in range(self.role_combo.count())]:
            self.role_combo.setCurrentText(self.role_md)
        self.role_combo.currentTextChanged.connect(self.on_role_changed)
        header.addWidget(self.role_combo, 1)
        header.addStretch()
        self.font_dec_btn = QPushButton("-")
        self.font_dec_btn.setFixedWidth(32)
        self.font_dec_btn.clicked.connect(lambda: self.apply_font_size(self.font_size - 1))
        self.font_inc_btn = QPushButton("+")
        self.font_inc_btn.setFixedWidth(32)
        self.font_inc_btn.clicked.connect(lambda: self.apply_font_size(self.font_size + 1))
        header.addWidget(self.font_dec_btn)
        header.addWidget(self.font_inc_btn)
        layout.addLayout(header)

        # Status row (spinner + label) shown while a file upload runs. Hidden when idle.
        self.status_row = QWidget()
        status_layout = QHBoxLayout(self.status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.upload_progress = QProgressBar()
        self.upload_progress.setRange(0, 0)
        self.upload_progress.setFixedWidth(140)
        self.upload_progress.setTextVisible(False)
        self.upload_status_label = QLabel("")
        status_layout.addWidget(self.upload_progress)
        status_layout.addWidget(self.upload_status_label, 1)
        self.status_row.setVisible(False)
        layout.addWidget(self.status_row)
        self.upload_worker = None

        # Text display area for messages (renders Markdown via setMarkdown)
        self.text_area = MarkdownTextEdit(self)
        layout.addWidget(self.text_area)

        # Display session header
        self.text_area.append(f"**Model:** `{self.model}` (Anthropic) — **Loaded:** {self.loaded_at}")
        self.text_area.append(f"**Agent:** {self.name} — **Role:** {self.role_md}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")

        # User input field
        self.user_input = QLineEdit(self)
        self.user_input.setPlaceholderText("Type your message and press Enter")
        layout.addWidget(self.user_input)

        # Button to copy latest assistant response
        self.copy_button = QPushButton("Copy Latest Answer")
        self.copy_button.clicked.connect(self.copy_latest_answer)
        layout.addWidget(self.copy_button)

        # Bind Enter key to user input processing
        self.user_input.returnPressed.connect(self.on_enter_pressed)

        # Apply layout to the window
        self.setLayout(layout)

        # Apply initial font size to everything
        self.apply_font_size(self.font_size)

    def _discover_md_files(self):
        # All .md files alongside this script, except the shared common.md.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(script_dir, "*.md")))
        return [f for f in files if f != "common.md"]

    def apply_font_size(self, size):
        self.font_size = max(6, min(48, int(size)))
        font = QFont()
        font.setPointSize(self.font_size)
        for child in self.findChildren(QWidget):
            child.setFont(font)
        # MarkdownTextEdit's document needs its default font set explicitly,
        # then re-rendered so the existing content adopts the new size.
        self.text_area.document().setDefaultFont(font)
        self.text_area.rerender()

    def on_role_changed(self, role):
        if not role or role == self.role_md:
            return
        self.role_md = role
        self.instructions = load_persona(self.common_md, self.role_md, {
            "user": self.user,
            "name": self.name,
        })
        # Reset conversation state so the new role isn't carrying old context.
        self.history = []
        self.text_area.clear()
        self.text_area.append(f"**Model:** `{self.model}` (Anthropic) — **Loaded:** {self.loaded_at}")
        self.text_area.append(f"**Agent:** {self.name} — **Role changed to:** {self.role_md}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept drag event if a file is present
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        # Handle file drop by extracting path
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.upload_file(file_path)

    def upload_file(self, file_path):
        """Drag-and-drop entry. Runs on a background QThread so the GUI shows
        a live spinner + status label instead of freezing."""
        if self.upload_worker is not None and self.upload_worker.isRunning():
            self.text_area.append(f"_(Already uploading; ignored drop of `{file_path}`)_")
            return

        filename = os.path.basename(file_path)
        model = self.model
        history = self.history
        client = self.client
        instructions = self.instructions
        max_tokens = self.max_tokens
        temperature = self.temperature

        def do_upload(status_cb):
            from file_loader import classify_file, read_text, read_base64

            status_cb("Classifying file (local)")
            category, mime_type = classify_file(file_path)

            if category == "text":
                status_cb("Reading text file (local)")
                text = read_text(file_path)
                ack_message = {
                    "role": "user",
                    "content": (
                        f"I've uploaded a text file '{filename}'. Contents:\n\n"
                        f"{text}\n\n"
                        "Please acknowledge this file. I will ask follow-up questions about it."
                    ),
                }
            elif category == "image":
                status_cb("Base64-encoding image (local)")
                b64_data = read_base64(file_path)
                ack_message = {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": mime_type, "data": b64_data,
                        }},
                        {"type": "text", "text": "Please acknowledge this file. I will ask follow-up questions about it."},
                    ],
                }
            elif category == "pdf":
                status_cb("Base64-encoding PDF (local)")
                b64_data = read_base64(file_path)
                ack_message = {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {
                            "type": "base64", "media_type": "application/pdf", "data": b64_data,
                        }},
                        {"type": "text", "text": "Please acknowledge this file. I will ask follow-up questions about it."},
                    ],
                }
            else:
                raise ValueError(
                    f"Unsupported file type: {mime_type or 'unknown'} ({filename}). "
                    "Supported: text (.txt/.md/.csv/.json/.ipynb/source code/…), "
                    "images (jpeg/png/gif/webp), PDFs."
                )

            history.append(ack_message)

            status_cb("Sending to Anthropic Claude (remote)")
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=instructions,
                messages=history,
                **_temperature_kwarg(model, temperature),
            )
            content = response.content[0].text
            history.append({"role": "assistant", "content": content})

            usage = None
            if getattr(response, "usage", None):
                inp = getattr(response.usage, "input_tokens", None)
                out = getattr(response.usage, "output_tokens", None)
                usage = {"input": inp, "output": out,
                         "total": (inp + out) if (inp is not None and out is not None) else None}
            return content, usage

        self.text_area.append(f"**Processing file:** `{filename}`")
        self.text_area.append(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        self._show_upload_status("Starting upload…")

        self.upload_worker = FileUploadWorker(do_upload)
        self.upload_worker.status.connect(self._on_upload_status)
        self.upload_worker.finished_ok.connect(self._on_upload_finished)
        self.upload_worker.finished_err.connect(self._on_upload_error)
        self.upload_worker.start()

    def _show_upload_status(self, msg):
        self.upload_status_label.setText(msg)
        self.status_row.setVisible(True)

    def _hide_upload_status(self):
        self.status_row.setVisible(False)
        self.upload_status_label.setText("")

    def _on_upload_status(self, msg):
        self._show_upload_status(msg)
        self.text_area.append(f"_…{msg}_")

    def _on_upload_finished(self, response, usage):
        usage_str = format_usage(usage) if usage else ""
        if usage_str:
            self.text_area.append(f"_Done — tokens: {usage_str}_")
        else:
            self.text_area.append("_Done._")
        self._hide_upload_status()
        self.latest_response = response
        self.text_area.append(f"{self.name}: {response}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def _on_upload_error(self, msg):
        self._hide_upload_status()
        self.text_area.append(f"**Upload error:** {msg}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def on_enter_pressed(self):
        # Process input when Enter is pressed
        user_input = self.user_input.text().strip()
        if user_input:
            self.process_user_input(user_input)
        self.user_input.clear()

    def process_user_input(self, user_input):
        # Display user input
        self.text_area.append(f"{self.user}: {user_input}")
        self.text_area.append(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.user_input.setEnabled(False)

        # Launch worker thread to handle assistant response
        self.worker_thread = LLMWorker(user_input, self)
        self.worker_thread.result_ready.connect(self.display_results)
        self.worker_thread.start()

    def display_results(self, response):
        # Show assistant response
        self.latest_response = response
        self.text_area.append(f"{self.name}: {response}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.user_input.setEnabled(True)

    def copy_latest_answer(self):
        # Copy latest assistant message to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(self.latest_response)
        self.text_area.append("Latest answer copied to clipboard.")


# Start the GUI application
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Claude-agent chat GUI.")
    parser.add_argument(
        "role_md",
        nargs="?",
        default="general.md",
        help="Path to a role markdown file (default: general.md). Examples: researcher.md, grant_writer_NIH.md, article_reviewer.md.",
    )
    args = parser.parse_args()

    app = QApplication([])
    chatbot = ClaudeChatbot(role_md_path=args.role_md)
    chatbot.show()
    app.exec_()
