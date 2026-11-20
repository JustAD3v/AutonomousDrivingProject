"""Main window: wires the tabs, terminal panel and subprocess worker together."""
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QSplitter, QTabWidget, QPushButton
)

from ui.worker import SubprocessWorker
from ui.terminal_panel import TerminalPanel
from ui.command_builder import build_command
from ui.tabs.mode_tab import ModeTab
from ui.tabs.eval_tab import EvalTab
from ui.tabs.config_tab import ConfigTab


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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.build_left_pane())

        self.terminal = TerminalPanel(self.current_dir)
        splitter.addWidget(self.terminal)
        splitter.setSizes([680, 600])

        main_layout.addWidget(splitter)

    def build_left_pane(self):
        group = QGroupBox("Options for conf")
        layout = QVBoxLayout(group)

        self.mode_tab = ModeTab()
        self.eval_tab = EvalTab()
        self.config_tab = ConfigTab()
        # A saved JSON config is applied automatically to the run settings.
        self.config_tab.config_saved.connect(self.mode_tab.set_config_path)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.mode_tab, "Operation & Model")
        self.tabs.addTab(self.eval_tab, "Evaluation & Rendering")
        self.tabs.addTab(self.config_tab, "Config JSON")
        layout.addWidget(self.tabs)

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
        layout.addLayout(btn_row)
        return group

    # ---- running a process -------------------------------------------------

    def execute_gui_command(self):
        if self.worker and self.worker.isRunning():
            self.terminal.append("[GUI] A process is already running. Stop it first.")
            return

        cmd = build_command(self.mode_tab.values(), self.eval_tab.values())
        self.start_worker(cmd)

    def start_worker(self, cmd):
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = SubprocessWorker(cmd, cwd=self.current_dir)
        self.worker.log_signal.connect(self.terminal.append)
        self.worker.finished_signal.connect(self.on_process_finished)
        self.worker.start()

    def stop_command(self):
        if self.worker:
            # One press is enough; disable so repeated clicks do nothing.
            self.stop_btn.setEnabled(False)
            self.terminal.append("\n[GUI] Terminating process...")
            self.worker.stop()

    def on_process_finished(self, code):
        self.terminal.append(f"\n[Exited with code {code}]")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # Don't leave an evaluation (and its render window) running behind us.
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()