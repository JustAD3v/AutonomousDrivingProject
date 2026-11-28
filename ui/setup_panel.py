from PyQt6.QtWidgets import QGroupBox, QFormLayout, QComboBox

from ui.widgets import PathInput


class SetupPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Setup", parent)
        form = QFormLayout(self)

        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["dqn", "sb3_dqn", "wocar_dqn", "ppo", "wocar_ppo", "sb3-ppo"]
        )
        self.config_input = PathInput(
            "Select JSON Config", start=".", file_filter="JSON Files (*.json)",
            placeholder="JSON config path",
        )

        form.addRow("Model Type:", self.model_combo)
        form.addRow("JSON Config (--config):", self.config_input)

    def apply_config(self, path, model_type):
        self.config_input.set_text(path)
        self.model_combo.setCurrentText(model_type)

    def values(self):
        return {
            "model_type": self.model_combo.currentText(),
            "config": self.config_input.text(),
        }
