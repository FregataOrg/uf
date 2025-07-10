# --- 1. Logger クラス ---
class Logger:
    """コンソールへのログ出力を一元管理するクラス。"""
    COLORS = {
        "HEADER": '\033[95m', "BLUE": '\033[94m', "GREEN": '\033[92m',
        "YELLOW": '\033[93m', "RED": '\033[91m', "ENDC": '\033[0m',
        "BOLD": '\033[1m', "UNDERLINE": '\033[4m'
    }
    def __init__(self):
        pass
    def _log(self, color: str, prefix: str, message: str):
        print(f"{color}{self.COLORS['BOLD']}{prefix}{self.COLORS['ENDC']} {message}")
    def info(self, message: str): self._log(self.COLORS['BLUE'], "[INFO]", message)
    def success(self, message: str): self._log(self.COLORS['GREEN'], "[SUCCESS]", f"✅ {message}")
    def warn(self, message: str): self._log(self.COLORS['YELLOW'], "[WARNING]", f"⚠️ {message}")
    def error(self, message: str): self._log(self.COLORS['RED'], "[ERROR]", f"❌ {message}")
    def start_section(self, title: str): print(f"\n{self.COLORS['HEADER']}{self.COLORS['BOLD']}--- {title} ---{self.COLORS['ENDC']}")