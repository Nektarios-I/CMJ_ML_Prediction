"""Compatibility entrypoint for Random Forest analysis plots.

This keeps the original filename while delegating to the refactored script.
"""

from scripts.analyze_random_forest import main


if __name__ == "__main__":
    main()
