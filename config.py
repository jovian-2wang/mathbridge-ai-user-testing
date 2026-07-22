from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CURRICULUM_DIR = DATA_DIR / "curriculum"
MEMORY_DIR = DATA_DIR / "memory"

APP_TITLE = "MathBridge AI"
APP_SUBTITLE = "Connected Math Learning Companion"

LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.3

# Palette from mockup
COLORS = {
    "blue": "#dbeafe",
    "blue_dark": "#2563eb",
    "green": "#dcfce7",
    "green_dark": "#166534",
    "yellow": "#fef9c3",
    "yellow_dark": "#a16207",
    "purple": "#ede9fe",
    "red": "#fee2e2",
    "border": "#d1d5db",
    "muted": "#6b7280",
}
