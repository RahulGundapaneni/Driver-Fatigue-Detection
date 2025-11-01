"""Legacy entry point kept for backward compatibility.

Prefer running `python detection.py` from the repository root. This wrapper is
provided so existing documentation that references `models/drowsiness_detection.py`
continues to work.
"""

from detection import main


if __name__ == "__main__":
    main()
