"""Small widgets shared between the panels."""
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog


class PathInput(QWidget):
    """A text field with a Browse button.

    mode: "open" (existing file), "save" (new file) or "dir" (folder).
    """

    def __init__(self, title, mode="open", start="models",
                 file_filter="All Files (*)", text="", placeholder="", parent=None):
        super().__init__(parent)
        self.title = title
        self.mode = mode
        self.start = start
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(text)
        self.edit.setPlaceholderText(placeholder)
        button = QPushButton("Browse...")
        button.clicked.connect(self.browse)
        layout.addWidget(self.edit)
        layout.addWidget(button)

    def browse(self):
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, self.title, self.start)
        elif self.mode == "save":
            path, _ = QFileDialog.getSaveFileName(
                self, self.title, self.start, self.file_filter
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, self.title, self.start, self.file_filter
            )
        if path:
            self.edit.setText(path)

    def text(self):
        return self.edit.text().strip()

    def set_text(self, text):
        self.edit.setText(text)


def seed_input():
    """Optional seed field that only accepts whole numbers."""
    edit = QLineEdit()
    edit.setValidator(QIntValidator(0, 999999999))
    edit.setPlaceholderText("Optional")
    return edit
