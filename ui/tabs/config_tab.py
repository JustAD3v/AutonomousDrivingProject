import os
import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox
)

from ui.config_schema import (
    ALGORITHMS, ENV_SECTION, FIELDS,
    sections, get_value, to_text, build_config,
)

ENVIRONMENTS = [
    "highway-fast-v0", "highway-v0", "merge-v0",
    "roundabout-v0", "parking-v0", "intersection-v0",
]


class ConfigTab(QWidget):
    # path of the saved file, algorithm it was made for
    config_saved = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.loaded_file = None  

        top = QFormLayout()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(ALGORITHMS.keys())
        self.algorithm_combo.textActivated.connect(self.show_current)

        button_row = QHBoxLayout()
        load_btn = QPushButton("Load JSON...")
        load_btn.clicked.connect(self.load_file)
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_row.addWidget(load_btn)
        button_row.addWidget(reset_btn)

        top.addRow("<b>Algorithm:</b>", self.algorithm_combo)
        top.addRow(button_row)
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

        self.show_current()

    # ---- loading -----------------------------------------------------------

    def show_current(self):
        if self.loaded_file:
            name, config = self.loaded_file
            self.show_config(config, f"File: {name}")
        else:
            self.show_config({}, "Program defaults")

    def reset_to_defaults(self):
        self.loaded_file = None
        self.show_current()

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
            name = os.path.basename(filename)
            self.show_config(config, f"File: {name}")
            self.loaded_file = (name, config)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file: {e}")

    def show_config(self, config, source):
        algorithm = self.algorithm_combo.currentText()
        content = QWidget()
        content_layout = QVBoxLayout(content)
        fields = {}
        env_group = None

        for title, keys in sections(algorithm).items():
            group = QGroupBox(title)
            form = QFormLayout(group)
            if title == ENV_SECTION:
                group.setTitle("Custom environment settings")
                group.setCheckable(True)
                group.setChecked(config.get("env_config") is not None)
                env_group = group

            for key in keys:
                kind = FIELDS[key][0]
                widget = self.make_widget(kind, get_value(config, key, algorithm))
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
            if kind in ("path", "opt_int"):
                widget.setPlaceholderText("Empty = none")
            elif kind == "seed":
                widget.setPlaceholderText("Empty = random")
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

            self.config_saved.emit(filename, self.algorithm_combo.currentText())
            QMessageBox.information(
                self,
                "Success",
                f"Config successfully saved to:\n{filename}\n\n"
                "The path and model type have been applied under Setup.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {e}")
