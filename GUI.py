import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import HighwayGUI


def main():
    app = QApplication(sys.argv)
    gui = HighwayGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
