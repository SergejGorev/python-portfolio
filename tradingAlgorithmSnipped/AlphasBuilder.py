from AlgorithmImports import *
from Alphas import *
import Config

from typing import Dict
from dateutil.relativedelta import relativedelta as rd
import pandas as pd
import api


class AlphasBuilder:
    def __init__(
        self,
        algo: QCAlgorithm,
        t0=None,
        t_1=1,
        t1=1,
    ) -> None:

        self.Algo = algo
        self.alphas = Config.ALPHAS

        self.t0 = self.Algo.Time if t0 is None else t0
        self.t1 = self.t0 + rd(months=t1)
        self.t_1 = self.t0 - rd(months=t_1)
        self.Schedule: pd.DataFrame = self.__initializeSchedule()
        self.count = {}

        self.AlphaInstances: Dict[str, AlphaFactor] = {}

    def __initializeSchedule(self):
        calendar = self.Algo.Dummy.GetDaysByType(
            TradingDayType.BusinessDay, self.t_1, self.t1
        )
        rIdx = pd.Index([i.Date.date() for i in calendar])
        cIdx = pd.MultiIndex.from_product([self.alphas, ["start", "end", "direction"]])
        schedule = pd.DataFrame(index=rIdx, columns=cIdx)
        self.Algo.Debug(
            f"Schedule initialized from {self.t_1.date()} to {self.t1.date()}"
        )
        return schedule

    def scheduleAlphas(self):
        alphasAdded = {}
        missingAlphas = []
        for alpha in self.alphas:
            instance: AlphaFactor = globals()[alpha](self.Algo)
            instance.scheduleAlpha(self.Schedule)

            self.AlphaInstances[alpha] = instance

            if self.Schedule[alpha].dropna().shape[0] > 0:
                alphasAdded[alpha] = self.Schedule[alpha].dropna().shape[0]
            else:
                missingAlphas.append(alpha)

        self.Algo.Debug(f"{len(alphasAdded)} from {len(self.alphas)} Alphas scheduled")
        self.Algo.Debug(alphasAdded)
        self.Algo.Debug(f"Missing Alphas: {missingAlphas}")

    def saveSchedule(self):
        res = {}

        for alpha in self.Schedule.columns.get_level_values(0).unique():
            tmp = self.Schedule[alpha].dropna()
            if tmp.empty:
                continue

            tmp = tmp[tmp["start"].dt.date >= dt.datetime.now().date()]
            tmp = tmp.dropna(axis=1)[["start", "end"]].reset_index(drop=True)
            if tmp.empty:
                continue

            res[alpha] = tmp.astype(str).to_dict("records")

        api.pushDummy(res)
