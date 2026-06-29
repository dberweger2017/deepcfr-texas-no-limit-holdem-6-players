# src/utils/settings.py
"""
Shared settings for the DeepCFR Poker AI project.
"""

# Global state validation settings
STRICT_CHECKING = False

def set_strict_checking(strict_mode):
    """Set the global strict checking mode."""
    global STRICT_CHECKING
    STRICT_CHECKING = bool(strict_mode)


def is_strict_checking():
    """Return the current strict checking mode."""
    return STRICT_CHECKING
