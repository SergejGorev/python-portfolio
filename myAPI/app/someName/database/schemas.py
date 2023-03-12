from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
    
class DummyEntry(BaseModel):
    user:str
    start:datetime
    end:datetime
    alpha:str

class TradeResponse(BaseModel):
    user:str
    alpha:str
    status:str

class DummyResponse(BaseModel):
    user:str
    status:str
    entriesPushed:int

class OpenTrade(BaseModel):
    user:str
    alpha:str
    quantity:float
    stopLoss:float
    active:Optional[bool]
    
class CloseTrade(BaseModel):
    user:str
    alpha:str

class TradeExists(BaseModel):
    user:str
    alpha:str