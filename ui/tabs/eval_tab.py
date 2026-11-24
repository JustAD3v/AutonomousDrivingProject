from PyQt6.QtWidgets import QWidget, QFormLayout, QCheckBox, QDoubleSpinBox


class EvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)

        self.render_chk = QCheckBox("Render environment during evaluation (--render_human)")
        self.render_chk.setChecked(True)

        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 1.0)
        self.delay_spin.setValue(0.05)
        self.delay_spin.setSingleStep(0.01)

        self.adversarial_chk = QCheckBox("Enable FGSM adversarial perturbation (--adversarial)")

        self.attack_bound_spin = QDoubleSpinBox()
        self.attack_bound_spin.setRange(0.01, 1.0)
        self.attack_bound_spin.setValue(0.2)
        self.attack_bound_spin.setSingleStep(0.05)

        form.addRow(self.render_chk)
        form.addRow("Render Delay Seconds (--render_delay):", self.delay_spin)
        form.addRow(self.adversarial_chk)
        form.addRow("Attack Bound (--attack_bound):", self.attack_bound_spin)

    def values(self):
        return {
            "render": self.render_chk.isChecked(),
            "render_delay": self.delay_spin.value(),
            "adversarial": self.adversarial_chk.isChecked(),
            "attack_bound": self.attack_bound_spin.value(),
        }
