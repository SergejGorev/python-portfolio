from fastapi import FastAPI, Depends
from myAPI.app.someName.routers import tradeBook, dummyStuff
from myAPI.app.auth.authBarer import JWTBearer

app = FastAPI(
    dependencies=[Depends(JWTBearer())]
)

app.include_router(tradeBook.router)
app.include_router(dummyStuff.router)