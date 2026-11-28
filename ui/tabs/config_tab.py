import os
import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox
)

from ui.config_schema import SECTIONS, has_key, get_value, to_text, build_config
from ui.config_templates import TEMPLATES

ENVIRONMENTS = [
    "highway-fast-v0", "highway-v0", "merge-v0",
    "roundabout-v0", "parking-v0", "intersection-v0",
]
ENV_SECTION = "Environment settings"


class ConfigTab(QWidget):
    config_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.addItems(TEMPLATES.keys())
        self.template_combo.textActivated.connect(self.load_template)
        load_btn = QPushButton("Load JSON...")
        load_btn.clicked.connect(self.load_file)
        top.addWidget(QLabel("<b>Template:</b>"))
        top.addWidget(self.template_combo, 1)
        top.addWidget(load_btn)
        layout.addLayout(top)

        self.source_label = QLabel()
        layout.addWidget(self.source_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        save_btn = QPushButton("Save && Apply JSON Config")
        save_btn.setMinimumHeight(35)
        save_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.save_config_json_file)
        layout.addWidget(save_btn)

        self.load_template(self.template_combo.currentText())

    # ---- loading -----------------------------------------------------------

    def load_template(self, name):
        self.show_config(TEMPLATES[name], f"Template: {name}")

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load JSON Config", ".", "JSON Files (*.json)"
        )
        if not filename:
            return
        try:
            with open(filename) as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("the file does not contain a JSON object")
            self.show_config(config, f"File: {os.path.basename(filename)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file: {e}")

    def show_config(self, config, source):
        """Rebuild the form, showing only the values this config uses."""
        content = QWidget()
        content_layout = QVBoxLayout(content)
        fields = {}
        env_group = None

        for title, entries in SECTIONS.items():
            if title == ENV_SECTION:
                # Unchecked means "env_config": null (the program's defaults).
                present = entries
                custom = config.get("env_config") is not None
            else:
                present = [entry for entry in entries if has_key(config, entry[0])]
            if not present:
                continue

            group = QGroupBox(title)
            form = QFormLayout(group)
            if title == ENV_SECTION:
                group.setTitle("Custom environment settings")
                group.setCheckable(True)
                group.setChecked(custom)
                env_group = group

            for key, kind in present:
                if has_key(config, key):
                    value = get_value(config, key)
                elif title == ENV_SECTION and not custom:
                    # Prefill so the user has sensible values if they enable it.
                    value = get_value(TEMPLATES["dqn"], key)
                else:
                    continue
                widget = self.make_widget(kind, value)
                fields[key] = widget
                form.addRow(key.split(".")[-1] + ":", widget)
            content_layout.addWidget(group)

        content_layout.addStretch()
        self.base_config = config
        self.fields = fields
        self.env_group = env_group
        self.source_label.setText(source)
        self.scroll.setWidget(content)

    def make_widget(self, kind, value):
        if kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
        elif kind == "env":
            widget = QComboBox()
            widget.setEditable(True)
            widget.addItems(ENVIRONMENTS)
            widget.setCurrentText(to_text(value))
        else:
            widget = QLineEdit(to_text(value))
            if kind == "path":
                widget.setPlaceholderText("Empty = none")
            elif kind.endswith("_list"):
                widget.setPlaceholderText("Comma separated, e.g. 256, 256")
        return widget

    # ---- saving ------------------------------------------------------------

    def read_fields(self):
        values = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            else:
                values[key] = widget.text()
        return values

    def save_config_json_file(self):
        try:
            config_data = build_config(
                self.base_config, self.read_fields(), self.env_group.isChecked()
            )
        except ValueError as e:
            QMessageBox.warning(self, "Invalid value", str(e))
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Config", "configs/custom_config.json", "JSON Files (*.json)"
        )
        if not filename:
            return
        if not filename.lower().endswith(".json"):
            filename += ".json"

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(config_data, f, indent=4)

            self.config_saved.emit(filename)
            QMessageBox.information(
                self,
                "Success",
                f"Config successfully saved to:\n{filename}\n\n"
                "It has been automatically applied to your run settings.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {e}")
