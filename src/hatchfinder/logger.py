
from pathlib import Path

class Logger:
    def __init__(self, path: Path, log_name: str = "log.txt") -> None:
        self.path = path
        self.log_name = log_name

        path.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        print(message)
        with open(self.path / self.log_name, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        