from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

HELP_HTML = """
<h2>Write help page later</h2>

"""


class HelpTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setHtml(HELP_HTML)
        layout.addWidget(browser)
