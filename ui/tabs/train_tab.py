from PyQt6.QtWidgets import QWidget, QFormLayout, QCheckBox

from ui.widgets import PathInput, seed_input


class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)

        self.save_input = PathInput(
            "Save Model As", mode="save", placeholder="e.g. models/my_model"
        )
        self.seed_input = seed_input()
        self.parallel_chk = QCheckBox("Parallel sample collection (--parallel)")
        self.sweep_chk = QCheckBox("Sweep runs, experimental (--sweep)")
        self.model_dir_input = PathInput(
            "Select Sweep Model Folder", mode="dir", placeholder="./models"
        )

        form.addRow("Save Model To (--net_file):", self.save_input)
        form.addRow("Random Seed (--seed):", self.seed_input)
        form.addRow(self.parallel_chk)
        form.addRow(self.sweep_chk)
        form.addRow("Sweep Model Folder (--model_dir):", self.model_dir_input)

        self.sweep_chk.toggled.connect(self.update_enabled)
        self.update_enabled()

    def update_enabled(self):
        # A sweep names its own model files and saves them in the folder.
        sweep = self.sweep_chk.isChecked()
        self.save_input.setEnabled(not sweep)
        self.model_dir_input.setEnabled(sweep)

    def values(self):
        return {
            "net_file": self.save_input.text(),
            "seed": self.seed_input.text().strip(),
            "parallel": self.parallel_chk.isChecked(),
            "sweep": self.sweep_chk.isChecked(),
            "model_dir": self.model_dir_input.text(),
        }
