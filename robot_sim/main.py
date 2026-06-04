"""
Entry point for the FANUC LR Mate 200iD/14L knife sharpening robot simulator.

Usage:
    python -m robot_sim.main
    # or from project root:
    python robot_sim/main.py
"""
import sys
import os

# Ensure package is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Launch the main GUI application."""
    try:
        from robot_sim.gui.main_window import MainWindow
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
