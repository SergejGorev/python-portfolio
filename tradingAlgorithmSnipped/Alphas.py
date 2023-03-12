from AlgorithmImports import *

import datetime as dt
import pytz
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta as rd

import pandas as pd
import math

from dummy import dates
import api

from random import randint

DUMMY_INT = randint(0, 23)
DUMMY_SYMBOL = Symbol.Create("SPY", SecurityType.Equity, Market.USA)
DUMMY_DIRECTION = InsightDirection.Up

class AlphaFactor:
    def __init__(self, algo) -> None:
        self.Name: str = ...
        self.Open: dt.time = ...
        self.Close: dt.time = ...
        self.Algo: QCAlgorithm = algo
        self.Symbol = ...
        self.Direction = InsightDirection.Flat
        self.Confidence: float = ...

        self._holdFromTo = []

        # Default Value, will be overriden by __initializeLoc
        self.locTz = self.Algo.TimeZone
        # Default Value, will be overriden by __initializeLoc
        self.locTime = self.Algo.Time

    def scheduleAlpha(self, schedule: pd.DataFrame):
        ...

    @property
    def holdFromTo(self):
        return self._holdFromTo

    @holdFromTo.setter
    def holdFromTo(self, value):
        self._holdFromTo = value

    def emitEntryInsight(self):
        self.Algo.EmitInsights(
            [
                Insight.Price(
                    self.Symbol,
                    dt.timedelta(days=1),
                    self.Direction,
                    sourceModel=f"#ENTRY#{self.Name}",
                    confidence=self.Confidence,
                )
            ]
        )

    def emitExitInsight(self):
        if api.tradeExists(self.Name):
            self.Algo.EmitInsights(
                [
                    Insight.Price(
                        self.Symbol,
                        dt.timedelta(days=1),
                        InsightDirection.Flat,
                        sourceModel=f"#EXIT#{self.Name}",
                        confidence=self.Confidence,
                    )
                ]
            )

    def __initializeLoc(self):
        self.locTz = pytz.timezone(
            str(self.Algo.Securities[self.Symbol].Exchange.TimeZone)
        )
        self.locTime = self.Algo.Securities[self.Symbol].LocalTime

    def dummy1(self):
        #### some data HARDFIX
        return [parse(d).date() for d in dates]

    def getWeekOfMonth(self, dt):
        adjustedDom = dt.day + dt.replace(day=1).weekday()
        return int(math.ceil(adjustedDom / 7.0))

    def getNextMarketClose(self):
        closeAt: dt = self.Algo.Securities[
            self.Symbol
        ].Exchange.Hours.GetNextMarketClose(self.locTime, False)
        return (
            self.locTz.localize(closeAt)
            .astimezone(pytz.timezone(str(self.Algo.TimeZone)))
            .replace(tzinfo=None)
        )

    def getNextMarketOpen(self):
        openAt: dt = self.Algo.Securities[self.Symbol].Exchange.Hours.GetNextMarketOpen(
            self.locTime, False
        )

        return (
            self.locTz.localize(openAt)
            .astimezone(pytz.timezone(str(self.Algo.TimeZone)))
            .replace(tzinfo=None)
        )

    def getHoldingPeriodRange(self, start: dt.date, end: dt.date, timAdj=10):
        self.__initializeLoc()
        openTrade = dt.datetime.combine(start, self.Open)
        if self.Name in ["InterestsOverWeekEnd", "ShortSummer"]:
            openTrade = (
                self.locTz.localize(openTrade)
                .astimezone(pytz.timezone(str(self.Algo.TimeZone)))
                .replace(tzinfo=None)
            )

        closeTrade = dt.datetime.combine(end, self.Close)

        if self.Name in ["InterestsOverWeekEnd", "ShortSummer"]:
            closeTrade = (
                self.locTz.localize(closeTrade)
                .astimezone(pytz.timezone(str(self.Algo.TimeZone)))
                .replace(tzinfo=None)
            )

        return openTrade, closeTrade


class SomeClass1(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(randint(0, 23), randint(0, 23))
        self.Close = dt.time(randint(0, 23), randint(0, 23))
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))

    def emitEntryInsight(self):
        if self.Algo.TTSMA.tradingBelow:
            super().emitEntryInsight()


class SomeClass2(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass3(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass4(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass5(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in self.dummy1():
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))

class SomeClass6(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in self.dummy1():
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass7(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in self.dummy1():
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))

class SomeClass8(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass9(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass10(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass11(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))

class SomeClass13(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))

class SomeClass14(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))


class SomeClass15(AlphaFactor):
    def __init__(self, algo) -> None:
        super().__init__(algo)
        self.Name = self.__class__.__name__
        self.Open = dt.time(DUMMY_INT, DUMMY_INT)
        self.Close = dt.time(DUMMY_INT, DUMMY_INT)
        self.Algo = algo
        self.Symbol = DUMMY_SYMBOL
        self.Direction = DUMMY_DIRECTION
        self.Confidence = DUMMY_INT

    def scheduleAlpha(self, Schedule: pd.DataFrame):
        for d in Schedule.index:
            holdingPeriod = Schedule.index.slice_indexer(d, DUMMY_INT)

            if len(Schedule.index[holdingPeriod]) < 1:
                break

            start, end = self.getHoldingPeriodRange(
                Schedule.index[holdingPeriod][0],
                Schedule.index[holdingPeriod][-1],
            )

            Schedule.loc[start.date(), (self.Name, "start")] = start
            Schedule.loc[start.date(), (self.Name, "end")] = end
            Schedule.loc[
                start.date(), (self.Name, "direction")
            ] = self.Direction
            self.holdFromTo.append((start, end))