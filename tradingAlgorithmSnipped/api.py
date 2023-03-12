import requests
import Config

USER = Config.CONFIG_FILE["USER"]
API_URL = Config.CONFIG_FILE["API_URL"]
API_TOKEN = Config.CONFIG_FILE["API_TOKEN"]

header = {"Authorization": "Bearer " + API_TOKEN}


class TradeRequest:
    def __init__(self, alpha: str, quantity: float, stopLoss: float):
        self.alpha = alpha
        self.quantity = quantity
        self.stopLoss = stopLoss

    def __dict__(self):
        return {
            "user": USER,
            "alpha": self.alpha,
            "quantity": self.quantity,
            "stopLoss": self.stopLoss,
        }


def tradeExists(alpha):
    body = {"user": USER, "alpha": alpha}
    return requests.get(f"{API_URL}/tradeExists", json=body, headers=header)


def getAllOpenTrades():
    return requests.get(f"{API_URL}/getAllOpenTrades/{USER}", headers=header)


def openTrade(trade):
    return requests.post(f"{API_URL}/openTrade", json=trade.__dict__(), headers=header)


def closeTrade(alpha):
    body = {"user": USER, "alpha": alpha}
    res = requests.post(f"{API_URL}/closeTrade", json=body, headers=header)
    return res


def pushDummy(dummy):
    body = []
    for alpha in dummy.keys():
        for entries in dummy[alpha]:
            entry = {
                "user": USER,
                "alpha": alpha,
                "start": entries["start"],
                "end": entries["end"],
            }
            body.append(entry)
    return requests.post(
        f"{API_URL}/tradingCalendar/push/{USER}", json=body, headers=header
    )
