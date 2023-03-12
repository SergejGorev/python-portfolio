from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from myAPI.app.someName.database import schemas, models, database
from myAPI.app.someName.dependencies import getDb
from typing import List
from fastapi.encoders import jsonable_encoder as encode


router = APIRouter(
    prefix="/dummy/dummy2",
    tags=["betterDummy"],
)

models.Base.metadata.create_all(bind=database.engine)


def deleteEntries(user:str,db: Session):
    db.query(models.Dummy).filter_by(user=user).delete()

@router.post("/push/{user}")
async def pushDummy(user:str, calendar: List[schemas.DummyEntry], db: Session=Depends(getDb)):
    count = 0
    for cal in calendar:
        c = models.Dummy(user=user,alpha=cal.alpha,start=cal.start,end=cal.end)
        db.add(c)
        count+=1

    db.commit()
    return encode(schemas.DummyResponse(user=user,status="success",entriesPushed=count))


@router.get("/get/{user}")
async def getDummy(user:str, db: Session=Depends(getDb)):
    return db.query(models.Dummy).filter_by(user=user).all()
