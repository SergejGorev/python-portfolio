from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from myAPI.app.someName.database import schemas, database, models
from myAPI.app.someName.dependencies import getDb
from fastapi.encoders import jsonable_encoder as encode


router = APIRouter(
    prefix="/dummy",
    tags=["veryGoodTag"],
)

models.Base.metadata.create_all(bind=database.engine)

def tradeFilter(trade):
    return {"active": True, "alpha": trade.alpha, "user": trade.user}

@router.get("/tradeExists")
async def tradeExists(trade: schemas.TradeExists, db: Session=Depends(getDb)):
    return bool(db.query(models.Trades).filter_by(**tradeFilter(trade)).first())

@router.post("/openTrade")
async def openTrade(trade: schemas.OpenTrade, db: Session=Depends(getDb)):
    if await tradeExists(trade,db):
        return encode(schemas.TradeResponse(user=trade.user, alpha=trade.alpha, status="trade exists"))
    openTrade = models.Trades(alpha=trade.alpha,user=trade.user,quantity=trade.quantity,stopLoss=trade.stopLoss)
    db.add(openTrade)
    db.commit()
    return encode(schemas.TradeResponse(user=openTrade.user, alpha=openTrade.alpha, status="successfully created"))

@router.post("/closeTrade")
async def closeTrade(trade: schemas.CloseTrade, db: Session=Depends(getDb)):
    closed = db.query(models.Trades).filter_by(**tradeFilter(trade)).first()
    if closed is None:
        return encode(schemas.TradeResponse(user=trade.user, alpha=trade.alpha,status="ether closed or not existend"))
    
    closed.active = False
    db.commit()
    return encode(schemas.TradeResponse(user=closed.user, alpha=closed.alpha,status="successfully closed"))

@router.get("/getAllOpenTrades/{user}")
async def getAllOpenTrades(user:str, db: Session=Depends(getDb)):
    return db.query(models.Trades).filter_by(active=True,user=user).all()