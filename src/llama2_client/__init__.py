import os
from pathlib import Path

def run():
    p_dir = Path(__file__).parent
    p_file = Path(p_dir, 'web-gui.py')
    if p_file.exists():
        os.system(f'streamlit run {p_file.resolve()}')
    else:
        raise ImportError('Cannot find web-gui.py file')