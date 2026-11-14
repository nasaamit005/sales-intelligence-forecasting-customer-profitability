import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


@pytest.fixture(scope="session")
def pipe():
    """Run the full pipeline once (deterministic) and share the outputs."""
    from salesintel import pipeline
    return pipeline.run(save=True, verbose=False)


@pytest.fixture(scope="session")
def df(pipe):
    return pipe["df"]


@pytest.fixture(scope="session")
def client(pipe):
    from web import create_app
    return create_app().test_client()
