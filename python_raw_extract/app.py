"""
Manga Extract & Clean — Entry Point
"""

import sys
import tkinter as tk
from tkinter import messagebox


def _check_dependencies():
    """Validate that core dependencies are importable."""
    try:
        import torch
        import cv2
        import numpy
        import manga_translator  # noqa: F401
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Dependency",
            f"A required dependency is missing.\n\n"
            f"Run: pip install -r requirements.txt\n\n"
            f"Error: {e}",
        )
        sys.exit(1)


def main():
    _check_dependencies()

    from gui import App

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
