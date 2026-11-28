from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser

HELP_HTML = """
<h2>Autonomous Driving DRL Studio</h2>
<p>Train and evaluate reinforcement-learning driving agents without typing
commands. The GUI builds a <code>src/main.py</code> command from your choices
and shows its output in the terminal panel.</p>

<h3>Getting started</h3>
<p>Choose the <b>Model Type</b> and, if needed, a <b>JSON Config</b> under
Setup. Both are used for training and evaluation. Then open a tab:</p>
<ul>
<li><b>Train:</b> choose where to save the model and press
<b>Start training</b>. Use a new file name to avoid overwriting a model.</li>
<li><b>Evaluate:</b> choose the model to load and press
<b>Start evaluation</b>. The model must match the model type.</li>
</ul>
<p><b>Stop</b> ends the running process. Only one process can run at a time.</p>
<p>The environment and continuous actions are set in the config. Without a
config, the default environment and discrete actions are used.</p>

<h3>Train</h3>
<ul>
<li><b>Random Seed:</b> optional, overrides the seed in the config. A seed of 0
is currently ignored by the program.</li>
<li><b>Parallel:</b> collect samples in parallel. This checkbox currently
overrides the config.</li>
<li><b>Sweep:</b> experimental, for testing only. Saves one model file per run
in the sweep folder, which must already exist (default ./models).</li>
</ul>

<h3>Evaluate</h3>
<ul>
<li><b>Random Seed:</b> optional. Evaluation does not use the seed in the
config. A seed of 0 is currently ignored.</li>
<li><b>Render environment:</b> show the simulation in a window. Closing the
window stops the evaluation.</li>
<li><b>Render Delay:</b> pause between steps, in seconds.</li>
<li><b>Adversarial perturbation:</b> disturbs what the agent observes, to test
how robust it is. The environment itself is not changed.</li>
<li><b>Attack Type:</b> <i>fgsm</i> or <i>random</i>.</li>
<li><b>Attack Bound:</b> the maximum size of the perturbation.</li>
</ul>

<h3>Config</h3>
<p>Create your own training configuration. Pick a <b>Template</b> or press
<b>Load JSON</b>, edit the values and press <b>Save &amp; Apply JSON Config</b>.
The saved file is filled in as the JSON Config path.</p>
<ul>
<li>Picking a template or loading a file replaces unsaved edits.</li>
<li>Only the values found in the template or file are shown.</li>
<li>Uncheck <i>Custom environment settings</i> to save no environment
settings.</li>
<li>A config does not select the model type. Choose a matching one under
Setup.</li>
</ul>

<h3>Troubleshooting</h3>
<p>Read the terminal panel for error messages. An exit code of 0 means the
process finished normally. Check that the model file matches the selected model
type and action space.</p>
"""


class HelpTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setHtml(HELP_HTML)
        layout.addWidget(browser)
