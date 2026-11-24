import os
import sys
import subprocess
import threading

from PyQt6.QtCore import QThread, pyqtSignal

# Seconds to wait after terminate() before escalating to a hard kill().
KILL_GRACE_SECONDS = 1.5


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
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env,
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
            timer = threading.Timer(KILL_GRACE_SECONDS, self._force_kill)
            timer.daemon = True
            timer.start()

    def _force_kill(self):
        if self.process and self.process.poll() is None:
            self.process.kill()
