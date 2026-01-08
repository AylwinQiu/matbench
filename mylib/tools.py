"""
Some utility tools for mylib.
"""
class Logger:
    def __init__(self, path):
        self.path = path
    def write(self, content):
        # TODO
        pass


from pathlib import Path
def find_project_root(start_path:str) -> str:
    current_path = Path(start_path).resolve()
    if current_path.is_file():
        current_path = current_path.parent
    while True:
        target_file = current_path / "pixi.toml"
        if target_file.exists():
            return str(current_path)
        if current_path.parent == current_path:
            raise Exception("Project root not found.")
        current_path = current_path.parent
    return None