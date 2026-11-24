from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
)


class TerminalPanel(QGroupBox):
    def __init__(self, cwd, parent=None):
        super().__init__("Terminal", parent)
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.cwd_label = QLabel(f"{cwd}")
        self.cwd_label.setStyleSheet("color: #666666; font-size: 11px;")
        clear_btn = QPushButton("Clear Terminal")
        clear_btn.clicked.connect(lambda: self.output.clear())
        top_bar.addWidget(self.cwd_label)
        top_bar.addStretch()
        top_bar.addWidget(clear_btn)
        layout.addLayout(top_bar)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.output.setStyleSheet("background-color: #121212; color: #00ff66;")
        self.output.append("Terminal ready.")
        layout.addWidget(self.output)

    def append(self, text):
        """Add a line and keep the view scrolled to the bottom."""
        self.output.append(text)
        scrollbar = self.output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
