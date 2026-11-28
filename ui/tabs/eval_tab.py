from PyQt6.QtWidgets import QWidget, QFormLayout, QCheckBox, QComboBox, QDoubleSpinBox

from ui.widgets import PathInput, seed_input


class EvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)

        self.load_input = PathInput(
            "Select Model File", text="models/targetNetwork_trained"
        )
        self.seed_input = seed_input()

        self.render_chk = QCheckBox("Render environment (--render_human)")
        self.render_chk.setChecked(True)

        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 1.0)
        self.delay_spin.setValue(0.05)
        self.delay_spin.setSingleStep(0.01)

        self.adversarial_chk = QCheckBox("Enable adversarial perturbation (--adversarial)")

        self.attack_type_combo = QComboBox()
        self.attack_type_combo.addItems(["fgsm", "random"])

        self.attack_bound_spin = QDoubleSpinBox()
        self.attack_bound_spin.setRange(0.01, 1.0)
        self.attack_bound_spin.setValue(0.2)
        self.attack_bound_spin.setSingleStep(0.05)

        form.addRow("Model To Load (--net_file):", self.load_input)
        form.addRow("Random Seed (--seed):", self.seed_input)
        form.addRow(self.render_chk)
        form.addRow("Render Delay Seconds (--render_delay):", self.delay_spin)
        form.addRow(self.adversarial_chk)
        form.addRow("Attack Type (--attack_type):", self.attack_type_combo)
        form.addRow("Attack Bound (--attack_bound):", self.attack_bound_spin)

        self.render_chk.toggled.connect(self.update_enabled)
        self.adversarial_chk.toggled.connect(self.update_enabled)
        self.update_enabled()

    def update_enabled(self):
        self.delay_spin.setEnabled(self.render_chk.isChecked())
        attack = self.adversarial_chk.isChecked()
        self.attack_type_combo.setEnabled(attack)
        self.attack_bound_spin.setEnabled(attack)

    def values(self):
        return {
            "net_file": self.load_input.text(),
            "seed": self.seed_input.text().strip(),
            "render": self.render_chk.isChecked(),
            "render_delay": self.delay_spin.value(),
            "adversarial": self.adversarial_chk.isChecked(),
            "attack_type": self.attack_type_combo.currentText(),
            "attack_bound": self.attack_bound_spin.value(),
        }
