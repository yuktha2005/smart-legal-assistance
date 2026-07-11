import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.connection import Base
from database.models import User, Document

@pytest.fixture(scope="module")
def db_session():
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

def test_create_user(db_session):
    user = User(username="testuser", email="test@example.com", password_hash="hashedpass")
    db_session.add(user)
    db_session.commit()
    
    saved_user = db_session.query(User).filter_by(username="testuser").first()
    assert saved_user is not None
    assert saved_user.email == "test@example.com"

def test_create_document(db_session):
    doc = Document(filename="test_doc.pdf", pages=5, chunk_count=20)
    db_session.add(doc)
    db_session.commit()
    
    saved_doc = db_session.query(Document).filter_by(filename="test_doc.pdf").first()
    assert saved_doc is not None
    assert saved_doc.pages == 5
