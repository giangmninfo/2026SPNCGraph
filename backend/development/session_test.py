from sqlalchemy import text
from backend.infrastructure.database.postgres import SessionLocal

def test_session():
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT 1"))
        print("SessionLocal works:", result.scalar())
    finally:
        db.close()

if __name__ == "__main__":
    test_session()
