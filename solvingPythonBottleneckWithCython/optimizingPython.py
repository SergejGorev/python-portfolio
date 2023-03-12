import engine
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from datetime import timedelta


class Direction(Enum):
    long = 1
    short = -1
    both = 0


@dataclass
class Trades:
    slName: str
    tpName: str
    params: str
    direction: str
    trades: pd.DataFrame


FUTURE = "CL"
COMMISSION = 1

SLIPPAGE = {
    "CL": 2,
    "NG": 4,
    "GC": 2,
    "E6": 3,
    "ES": 2,
    "NQ": 4,
    "YM": 4,
}

MIN_TICK = {
    "CL": 0.01,
    "NG": 0.001,
    "GC": 0.01,
    "E6": 0.00005,
    "ES": 0.25,
    "NQ": 0.25,
    "YM": 1,
}

TICK_VALUE = {
    "CL": 10,
    "NG": 10,
    "GC": 10,
    "E6": 6.25,
    "ES": 12.5,
    "NQ": 5,
    "YM": 5,
}


def getTradesCythonEngine(
    SL: pd.Series,
    TP: pd.Series,
    signals: pd.Index,
    params: str,
    direction: Direction,
    df: pd.DataFrame,
) -> Trades:
    signalIndexer = df.index.get_indexer_for(signals)
    entries = df.iloc[np.add(signalIndexer, 1)]["o"].values

    # print(sl.shape,tp.shape)

    sl = np.subtract(entries, np.multiply(SL.loc[signals], direction.value))
    tp = np.add(entries, np.multiply(TP.loc[signals], direction.value))

    trades = engine.run(
        signalIndexer,
        df["h"].values,
        df["l"].values,
        sl.values,
        tp.values,
        direction.value,
    )
    if trades.size == 0:
        return None
    t1 = pd.Index(df.index[trades])

    trades = pd.DataFrame()
    trades["oTime"] = signals
    trades["cTime"] = t1
    trades["entry"] = entries
    trades["SL"] = SL.loc[signals].values
    trades["TP"] = TP.loc[signals].values
    trades["exit"] = np.where(df.loc[t1, "h"].values >= sl.values, sl.values, tp.values)
    trades["duration"] = (trades["cTime"] - trades["oTime"]) // timedelta(minutes=1)
    trades["pl"] = (
        trades["exit"].sub(trades["entry"]).div(MIN_TICK[FUTURE]).mul(direction.value)
    )
    trades["plNet"] = trades["pl"].sub(SLIPPAGE[FUTURE]).sub(COMMISSION)
    return Trades(SL.name, TP.name, params, direction.name, trades.set_index("oTime"))
