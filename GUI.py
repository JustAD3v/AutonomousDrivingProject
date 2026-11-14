import os
import sys
import json
import subprocess
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QLineEdit, QPushButton,
    QTextEdit, QSplitter, QTabWidget, QCheckBox, QDoubleSpinBox, 
    QSpinBox, QFileDialog, QLabel, QMessageBox
)

# ======== 1. Background Worker to Run CLI Commands Without Freezing UI ==========
class SubprocessWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, cmd, cwd=None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd or os.getcwd()
        self.process = None

    def run(self):
        display_cmd = " ".join(self.cmd)
        self.log_signal.emit(f"Executing: {display_cmd}\n")
        
        env = os.environ.copy()
        venv_bin = os.path.dirname(sys.executable)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env
            )
            for line in self.process.stdout:
                self.log_signal.emit(line.rstrip())

            self.process.wait()
            self.finished_signal.emit(self.process.returncode)
        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(-1)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


# ====== 2. Main Studio Window =====
class HighwayGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autonomous Driving DRL Studio")
        self.resize(1280, 700)
        self.worker = None
        self.current_dir = os.getcwd()

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ================= LEFT PANE =================
        conf_group = QGroupBox("Options for conf")
        conf_layout = QVBoxLayout(conf_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_mode_tab(), "Operation & Model")
        self.tabs.addTab(self.build_eval_tab(), "Evaluation & Rendering")
        self.tabs.addTab(self.build_config_tab(), "Config JSON")
        conf_layout.addWidget(self.tabs)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Process")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.clicked.connect(self.execute_gui_command)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_command)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        conf_layout.addLayout(btn_row)

        # ================= RIGHT PANE (OUTPUT TERMINAL) =================
        terminal_group = QGroupBox("Terminal")
        terminal_layout = QVBoxLayout(terminal_group)

        top_term_bar = QHBoxLayout()
        self.cwd_label = QLabel(f"{self.current_dir}")
        self.cwd_label.setStyleSheet("color: #666666; font-size: 11px;")
        clear_btn = QPushButton("Clear Terminal")
        clear_btn.clicked.connect(lambda: self.terminal_output.clear())
        top_term_bar.addWidget(self.cwd_label)
        top_term_bar.addStretch()
        top_term_bar.addWidget(clear_btn)
        terminal_layout.addLayout(top_term_bar)

        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Consolas", 10))
        self.terminal_output.setStyleSheet("background-color: #121212; color: #00ff66;")
        self.terminal_output.append("Terminal ready.")
        terminal_layout.addWidget(self.terminal_output)

        h_splitter.addWidget(conf_group)
        h_splitter.addWidget(terminal_group)
        h_splitter.setSizes([680, 600])

        main_layout.addWidget(h_splitter)

    def build_mode_tab(self):
        widget = QWidget()
        form = QFormLayout(widget)

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
        return widget

    def build_eval_tab(self):
        widget = QWidget()
        form = QFormLayout(widget)

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
        return widget

    def build_config_tab(self):
        widget = QWidget()
        form = QFormLayout(widget)

        self.json_env_combo = QComboBox()
        self.json_env_combo.addItems([
            "highway-fast-v0", 
            "highway-v0", 
            "merge-v0", 
            "roundabout-v0", 
            "parking-v0",
            "intersection-v0"
        ])

        self.json_iterations_spin = QSpinBox()
        self.json_iterations_spin.setRange(10, 100000)
        self.json_iterations_spin.setValue(500)
        self.json_iterations_spin.setSingleStep(50)

        self.json_lr_spin = QDoubleSpinBox()
        self.json_lr_spin.setDecimals(6)
        self.json_lr_spin.setRange(0.000001, 1.0)
        self.json_lr_spin.setValue(0.0005)
        self.json_lr_spin.setSingleStep(0.0001)

        self.json_seed_spin = QSpinBox()
        self.json_seed_spin.setRange(0, 999999)
        self.json_seed_spin.setValue(123)

        save_json_btn = QPushButton(" Save & Apply JSON Config")
        save_json_btn.setMinimumHeight(35)
        save_json_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        save_json_btn.clicked.connect(self.save_config_json_file)

        form.addRow(QLabel("<b>Target Environment:</b>"), self.json_env_combo)
        form.addRow("Iterations / Epochs:", self.json_iterations_spin)
        form.addRow("Learning Rate:", self.json_lr_spin)
        form.addRow("Random Seed:", self.json_seed_spin)
        form.addRow(save_json_btn)
        return widget

    def browse_net_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint / Network File", "models", "All Files (*)")
        if filename:
            self.net_input.setText(filename)

    def browse_config_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select JSON Config", ".", "JSON Files (*.json)")
        if filename:
            self.config_input.setText(filename)

    def save_config_json_file(self):
        config_data = {
            "env_id": self.json_env_combo.currentText(),
            "iterations": self.json_iterations_spin.value(),
            "learning_rate": self.json_lr_spin.value(),
            "seed": self.json_seed_spin.value()
        }

        filename, _ = QFileDialog.getSaveFileName(self, "Save JSON Config", "configs/custom_config.json", "JSON Files (*.json)")
        if filename:
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    json.dump(config_data, f, indent=4)
                
                self.config_input.setText(filename)
                QMessageBox.information(self, "Success", f"Config successfully saved to:\n{filename}\n\nIt has been automatically applied to your run settings.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")

    def execute_gui_command(self):
        if self.worker and self.worker.isRunning():
            self.append_terminal("[GUI] A process is already running. Stop it first.")
            return

        cmd = [sys.executable, "src/main.py"]

        if self.train_chk.isChecked():
            cmd.append("--train")

        cmd.extend(["--model_type", self.model_combo.currentText()])

        net_path = self.net_input.text().strip()
        if net_path:
            cmd.extend(["--net_file", net_path])

        config_path = self.config_input.text().strip()
        if config_path:
            cmd.extend(["--config", config_path])

        seed_val = self.seed_input.text().strip()
        if seed_val:
            cmd.extend(["--seed", seed_val])

        if not self.train_chk.isChecked():
            if self.render_chk.isChecked():
                cmd.append("--render_human")
                cmd.extend(["--render_delay", str(self.delay_spin.value())])
            if self.adversarial_chk.isChecked():
                cmd.append("--adversarial")
                cmd.extend(["--attack_bound", str(self.attack_bound_spin.value())])

        self.start_worker(cmd)

    def start_worker(self, cmd):
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = SubprocessWorker(cmd, cwd=self.current_dir)
        self.worker.log_signal.connect(self.append_terminal)
        self.worker.finished_signal.connect(self.on_process_finished)
        self.worker.start()

    def stop_command(self):
        if self.worker:
            self.append_terminal("\n[GUI] Terminating process...")
            self.worker.stop()

    def append_terminal(self, text):
        self.terminal_output.append(text)
        scrollbar = self.terminal_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_process_finished(self, code):
        self.append_terminal(f"\n[Exited with code {code}]")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = HighwayGUI()
    gui.show()
    sys.exit(app.exec())
