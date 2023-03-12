from AlgorithmImports import *
import numpy as np
import api


class Trade:
    def __init__(
        self,
        symbol,
        alpha,
        action="",
        quantity: float = 0,
        stop=None,
        direction=None,
        stopLoss=None,
    ) -> None:
        self.symbol: Symbol = symbol
        self.alpha = alpha
        self.action = action
        self.quantity = quantity
        self.stop = stop
        self.direction = direction
        self.stopLoss = stopLoss

    @property
    def isOpen(self) -> bool:
        return api.tradeExists(self.alpha)

    def __str__(self):
        return f"{self.alpha} is holding {self.quantity} units of {self.symbol}"


def getOpenTrades(algo):
    trades = api.getAllOpenTrades().json()
    res = {}
    if not trades:
        return res
    for t in trades:
        trade = Trade(
            symbol=algo.alphaSymbolMap[t["alpha"]],
            alpha=t["alpha"],
            quantity=t["quantity"],
            direction=getDirection(np.sign(-1 * t["quantity"])),
            stopLoss=t["stopLoss"],
        )
        res[trade.alpha] = trade
    return res


def getDirection(direction):
    return (
        InsightDirection.Up
        if direction > 0
        else InsightDirection.Down
        if direction < 0
        else InsightDirection.Flat
    )
