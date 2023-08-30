import os
from pathlib import Path


def run():
    p_dir = Path(__file__).parent
    script_file = 'web_gui.py'
    p_file = Path(p_dir, script_file)
    if p_file.exists():
        os.system(f'streamlit run {p_file.resolve()}')
    else:
        raise ImportError(f'Cannot find {script_file} file')
