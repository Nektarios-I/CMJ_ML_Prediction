"""Compatibility entrypoint for training all models.

This keeps the original filename while delegating to the refactored script.
"""

from scripts.train_all_models import main


if __name__ == "__main__":
    main()
