"""Launch Nivo web UI."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    ui_path = Path(__file__).parent / "src" / "ui" / "web_ui.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])
