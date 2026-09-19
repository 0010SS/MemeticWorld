"""Tests always run offline: mock LLM, hash embeddings, throwaway database."""
import os
import tempfile

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBED_PROVIDER"] = "hash"
os.environ["MEMETIC_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="memeticworld-test-"), "test.db")

import pytest  # noqa: E402

from app.db import database as db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "run.db"))
