import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists

from app.api.deps import get_db_session
from app.config import settings
from app.db.base import BaseModel
from app.main import app

os.environ["ENVIRONMENT"] = "testing"
TEST_DATABASE_URL = settings.DATABASE_URL + "_test"

if not database_exists(TEST_DATABASE_URL):
    create_database(TEST_DATABASE_URL)

engine = create_engine(TEST_DATABASE_URL)


TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

BaseModel.metadata.create_all(bind=engine)


@pytest.fixture(scope="session")
def db_engine():
    BaseModel.metadata.drop_all(bind=engine)
    BaseModel.metadata.create_all(bind=engine)
    yield engine
    BaseModel.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db_session] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    del app.dependency_overrides[get_db_session]
