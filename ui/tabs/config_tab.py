import os
import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QLabel, QFileDialog, QMessageBox
)


class ConfigTab(QWidget):
    config_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)

        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "highway-fast-v0",
            "highway-v0",
            "merge-v0",
            "roundabout-v0",
            "parking-v0",
            "intersection-v0",
        ])

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(10, 100000)
        self.iterations_spin.setValue(500)
        self.iterations_spin.setSingleStep(50)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setValue(0.0005)
        self.lr_spin.setSingleStep(0.0001)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(123)

        save_btn = QPushButton(" Save & Apply JSON Config")
        save_btn.setMinimumHeight(35)
        save_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.save_config_json_file)

        form.addRow(QLabel("<b>Target Environment:</b>"), self.env_combo)
        form.addRow("Iterations / Epochs:", self.iterations_spin)
        form.addRow("Learning Rate:", self.lr_spin)
        form.addRow("Random Seed:", self.seed_spin)
        form.addRow(save_btn)

    def save_config_json_file(self):
        config_data = {
            "env_id": self.env_combo.currentText(),
            "iterations": self.iterations_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "seed": self.seed_spin.value(),
        }

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Config", "configs/custom_config.json", "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
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
            QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
