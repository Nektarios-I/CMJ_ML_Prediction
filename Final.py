"""Compatibility entrypoint for tuned Random Forest training.

This keeps the original filename while delegating to the refactored script.
"""

from scripts.train_random_forest import main


if __name__ == "__main__":
    main()
