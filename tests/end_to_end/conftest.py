import os

import pytest


@pytest.fixture
def fixtures_path():
    """Return the path containing the standard end-to-end fixtures."""
    return os.path.join(os.path.dirname(__file__), "fixtures")
