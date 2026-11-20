"""'Operation & Model' tab: train/eval switch, model type, checkpoint, config, seed."""
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QCheckBox, QComboBox,
    QLineEdit, QPushButton, QFileDialog
)


class ModeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)

        self.train_chk = QCheckBox("Train model (Enables --train)")
        self.train_chk.setChecked(False)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["dqn", "sb3_dqn", "wocar_dqn", "ppo", "sb3-ppo"])

        net_layout = QHBoxLayout()
        self.net_input = QLineEdit("models/targetNetwork_trained")
        net_btn = QPushButton("Browse...")
        net_btn.clicked.connect(self.browse_net_file)
        net_layout.addWidget(self.net_input)
        net_layout.addWidget(net_btn)

        config_layout = QHBoxLayout()
        self.config_input = QLineEdit("")
        self.config_input.setPlaceholderText("Optional JSON config path")
        config_btn = QPushButton("Browse...")
        config_btn.clicked.connect(self.browse_config_file)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(config_btn)

        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("123")

        form.addRow(self.train_chk)
        form.addRow("Model Type:", self.model_combo)
        form.addRow("Checkpoint Path (--net_file):", net_layout)
        form.addRow("JSON Config (--config):", config_layout)
        form.addRow("Random Seed (--seed):", self.seed_input)

    def browse_net_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint / Network File", "models", "All Files (*)"
        )
        if filename:
            self.net_input.setText(filename)

    def browse_config_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Config", ".", "JSON Files (*.json)"
        )
        if filename:
            self.config_input.setText(filename)

    def set_config_path(self, path):
        """Called when the Config JSON tab saves a file."""
        self.config_input.setText(path)

    def values(self):
        return {
            "train": self.train_chk.isChecked(),
            "model_type": self.model_combo.currentText(),
            "net_file": self.net_input.text().strip(),
            "config": self.config_input.text().strip(),
            "seed": self.seed_input.text().strip(),
        }