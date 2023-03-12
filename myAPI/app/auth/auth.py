from jose import jwt
from decouple import config
from typing import Dict

JWT_SECRET = config("secret")
JWT_ALGORITHM = config("algorithm")



def token_response(token: str):
    return {
        "access_token": token
    }


def decodeJWT(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])