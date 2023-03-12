from sqlalchemy import Boolean, Column, Integer, String, DateTime, Float
from .database import Base

    
class Dummy(Base):
    __tablename__ = "dummy"
    
    id = Column(Integer, primary_key=True, index=True)
    a = Column(String)
    b = Column(DateTime)
    c = Column(DateTime)
    d = Column(String)

class Trades(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    user = Column(String)
    alpha = Column(String)
    quantity = Column(Float)
    stopLoss = Column(Float)
    active = Column(Boolean, default=True)
    