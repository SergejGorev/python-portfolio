from myAPI.app.someName.database import database
def getDb():
    try:
        db = database.SessionLocal()
        yield db
    finally:
        db.close()