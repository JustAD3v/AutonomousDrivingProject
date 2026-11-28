"""Main window: wires the setup panel, tabs, terminal panel and subprocess worker together."""
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QSplitter, QTabWidget, QPushButton
)

from ui.worker import SubprocessWorker
from ui.terminal_panel import TerminalPanel
from ui.command_builder import build_train_command, build_eval_command
from ui.setup_panel import SetupPanel
from ui.tabs.train_tab import TrainTab
from ui.tabs.eval_tab import EvalTab
from ui.tabs.config_tab import ConfigTab
from ui.tabs.help_tab import HelpTab


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
        group = QGroupBox("Options")
        layout = QVBoxLayout(group)

        self.setup_panel = SetupPanel()
        layout.addWidget(self.setup_panel)

        self.train_tab = TrainTab()
        self.eval_tab = EvalTab()
        self.config_tab = ConfigTab()
        # A saved JSON config is applied automatically to the setup.
        self.config_tab.config_saved.connect(self.setup_panel.set_config_path)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.eval_tab, "Evaluate")
        self.tabs.addTab(self.config_tab, "Config")
        self.tabs.addTab(HelpTab(), "Help")
        layout.addWidget(self.tabs)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton()
        self.run_btn.setMinimumHeight(38)
        self.run_btn.clicked.connect(self.execute_gui_command)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_command)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.tabs.currentChanged.connect(self.update_run_button)
        self.tabs.setCurrentWidget(self.eval_tab)
        self.update_run_button()
        return group

    # ---- running a process -------------------------------------------------

    def is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def update_run_button(self):
        """The open tab decides what the run button does."""
        tab = self.tabs.currentWidget()
        if tab is self.train_tab:
            text = "Start training"
        elif tab is self.eval_tab:
            text = "Start evaluation"
        else:
            text = "Open Train or Evaluate to run"
        self.run_btn.setText(text)
        can_run = tab in (self.train_tab, self.eval_tab)
        self.run_btn.setEnabled(can_run and not self.is_running())

    def execute_gui_command(self):
        if self.is_running():
            self.terminal.append("[GUI] A process is already running. Stop it first.")
            return

        setup = self.setup_panel.values()
        try:
            if self.tabs.currentWidget() is self.train_tab:
                cmd = build_train_command(setup, self.train_tab.values())
            else:
                cmd = build_eval_command(setup, self.eval_tab.values())
        except ValueError as e:
            self.terminal.append(f"[GUI] {e}")
            return
        self.start_worker(cmd)

    def start_worker(self, cmd):
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = SubprocessWorker(cmd, cwd=self.current_dir)
        self.worker.log_signal.connect(self.terminal.append)
        self.worker.finished_signal.connect(self.on_process_finished)
        # finished fires once the thread has really stopped.
        self.worker.finished.connect(self.update_run_button)
        self.worker.start()

    def stop_command(self):
        if self.worker:
            # One press is enough; disable so repeated clicks do nothing.
            self.stop_btn.setEnabled(False)
            self.terminal.append("\n[GUI] Terminating process...")
            self.worker.stop()

    def on_process_finished(self, code):
        self.terminal.append(f"\n[Exited with code {code}]")
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # Don't leave an evaluation (and its render window) running behind us.
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()