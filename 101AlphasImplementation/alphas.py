import time, sys
from dataclasses import dataclass
import pandas as pd, numpy as np
from numba import jit
import matplotlib.pyplot as plt, seaborn as sns
import datetime as dt



def report_progress(job_num, num_jobs, time0, task):
    """
    Advances in Financial Machine Learning, Snippet 20.9.1, pg 312.

    Example of Asynchronous call to pythons multiprocessing library

    :param job_num: (int) Number of current job
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    :return: (None)
    """
    # Report progress as asynch jobs are completed
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))

    msg = (
        time_stamp
        + " "
        + str(round(msg[0] * 100, 2))
        + "% "
        + task
        + " done after "
        + str(round(msg[1], 2))
        + " minutes. Remaining "
        + str(round(msg[2], 2))
        + " minutes."
    )

    if job_num < num_jobs:
        sys.stderr.write(msg + "\r")
    else:
        sys.stderr.write(msg + "\n")


class FeatureUtils:
    @staticmethod
    @jit
    def np_vwap(high, low, vol):
        return np.cumsum(vol * (high + low) / 2) / np.cumsum(vol)

    @staticmethod
    def pctRank(arr):
        return (arr.argsort().argmax() + 1) / arr.shape[0]

    @staticmethod
    def linearDecay(arr):
        weight = np.arange(arr.shape[0]) + 1
        weight = weight / weight.sum()
        return arr.T.dot(weight)

    @staticmethod
    def rollingWindow(arr, w):
        shape = arr.shape[:-1] + (arr.shape[-1] - w + 1, w)
        strides = arr.strides + (arr.strides[-1],)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    @staticmethod
    def reindex(arr, idx):
        arr = (
            pd.Series(arr)
            .sort_index(ascending=False)
            .reset_index(drop=True)
            .reindex(pd.RangeIndex(0, idx.shape[0], 1))
            .sort_index(ascending=False)
        )
        arr.index = idx
        return arr

    @staticmethod
    def getRollingValuesVectorized(series, w, func):
        return FeatureUtils.reindex(
            np.apply_along_axis(func, 1, FeatureUtils.rollingWindow(series.values, w)),
            series.index,
        )

    @staticmethod
    def normalize(series):
        return (series - series.mean()) / series.std()


@dataclass
class FormulaicAlphas101:
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    vol: pd.Series

    rollingLag: int = 1
    diffLag: int = 1
    shiftLag: int = 1
    otherLag: int = 1

    def __post_init__(self):
        self.ret = self.close.pct_change()
        self.vwap = pd.Series(
            FeatureUtils.np_vwap(self.high.values, self.low.values, self.vol.values),
            index=self.close.index,
        )

    def rollingValues(self, series, w, func):
        return FeatureUtils.getRollingValuesVectorized(series, w, func)

    def normalize(self, series):
        return FeatureUtils.normalize(series)

    def linearDecay(self, arr):
        return FeatureUtils.linearDecay(arr)

    def pctRank(self, arr):
        return FeatureUtils.pctRank(arr)

    def adv(self, v):
        return self.vol.rolling(v).mean()

    def alpha1(self):
        signedPower = np.square(
            np.where(
                self.ret < 0,
                self.ret.rolling(20 * self.rollingLag).std(skipna=False),
                self.close,
            )
        )
        return (
            self.rollingValues(
                pd.Series(signedPower, index=self.close.index),
                5 * self.rollingLag,
                np.argmax,
            ).rank(pct=True)
            - 0.5
        )

    def alpha2(self):
        return -1 * pd.Series(np.log(self.vol).diff(2 * self.diffLag)).rank(
            pct=True
        ).rolling(6 * self.rollingLag).corr(
            ((self.close - self.open).div(self.open)).rank(pct=True)
        )

    def alpha3(self):
        return -1 * self.open.rank(pct=True).rolling(10 * self.rollingLag).corr(
            self.vol.rank(pct=True)
        )

    def alpha4(self):
        return -1 * self.rollingValues(
            self.low.rank(pct=True), 9 * self.rollingLag, self.pctRank
        )

    def alpha5(self):
        return (
            (self.open - self.vwap.rolling(10 * self.rollingLag).mean()).rank(pct=True)
            * -1
            * (self.close - self.vwap).rank(pct=True).abs()
        )

    def alpha6(self):
        return -1 * self.open.rolling(10 * self.rollingLag).corr(self.vol)

    def alpha7(self):
        c = self.adv(20) < self.vol.values
        return pd.Series(
            np.where(
                c,
                -1
                * self.rollingValues(
                    self.close.diff(7 * self.diffLag).abs(),
                    60 * self.rollingLag,
                    self.pctRank,
                ).mul(np.sign(self.close.diff(7 * self.diffLag))),
                -1,
            ),
            index=self.close.index,
        )

    def alpha8(self):
        tmp = (
            self.open.rolling(5 * self.rollingLag).sum()
            * self.ret.rolling(5 * self.rollingLag).sum()
        )
        return -1 * (tmp - tmp.shift(10 * self.shiftLag)).rank(pct=True)

    def alpha9(self):
        delta = self.close.diff(1 * self.diffLag)
        return pd.Series(
            np.where(
                0 < delta.rolling(5 * self.rollingLag).min(),
                delta,
                np.where(
                    delta.rolling(5 * self.rollingLag).max() < 0, delta, -1 * delta
                ),
            ),
            index=self.close.index,
        )

    def alpha10(self):
        delta = self.close.diff(1 * self.shiftLag)
        return pd.Series(
            np.where(
                0 < delta.rolling(4 * self.rollingLag).min(),
                delta,
                np.where(
                    delta.rolling(4 * self.rollingLag).max() < 0, delta, -1 * delta
                ),
            ),
            index=self.close.index,
        ).rank(pct=True)

    def alpha11(self):
        return (
            (self.vwap - self.close).rolling(3 * self.rollingLag).max().rank(pct=True)
            + (self.vwap - self.close).rolling(3 * self.rollingLag).min().rank(pct=True)
        ) * self.vol.diff(3 * self.diffLag).rank(pct=True)

    def alpha12(self):
        return np.sign(self.vol.diff(1 * self.diffLag)) * (
            -1 * self.close.diff(1 * self.diffLag)
        )

    def alpha13(self):
        return -1 * self.close.rank(pct=True).rolling(5 * self.rollingLag).cov(
            self.vol.rank(pct=True)
        ).rank(pct=True)

    def alpha14(self):
        return (
            -1 * self.ret.diff(3 * self.diffLag).rank(pct=True)
        ) * self.open.rolling(10 * self.rollingLag).corr(self.vol)

    def alpha15(self):
        return (
            -1
            * self.high.rank(pct=True)
            .rolling(3 * self.rollingLag)
            .corr(self.vol.rank(pct=True))
            .rank(pct=True)
            .rolling(3 * self.rollingLag)
            .sum()
        )

    def alpha16(self):
        return -1 * self.high.rank(pct=True).rolling(5 * self.rollingLag).cov(
            self.vol.rank(pct=True)
        ).rank(pct=True)

    def alpha17(self):
        return (
            (
                -1
                * self.rollingValues(
                    self.close, 10 * self.rollingLag, self.pctRank
                ).rank(pct=True)
            )
            * self.close.diff(1 * self.diffLag).diff(1 * self.diffLag).rank(pct=True)
            * self.rollingValues(
                self.vol.div(self.adv(20)), 5 * self.rollingLag, self.pctRank
            ).rank(pct=True)
        )

    def alpha18(self):
        return -1 * (
            (self.close - self.open)
            .abs()
            .rolling(5 * self.rollingLag)
            .std()
            .rank(pct=True)
            + (self.close - self.open)
            + self.close.rolling(10 * self.rollingLag).corr(self.open)
        ).rank(pct=True)

    def alpha19(self):
        return (
            -1
            * np.sign(
                self.close
                - self.close.shift(7 * self.shiftLag)
                + self.close.diff(7 * self.diffLag)
            )
        ) * (
            1
            + (1 + self.close.pct_change().rolling(250 * self.rollingLag).sum()).rank(
                pct=True
            )
        )

    def alpha19GiveItATry(self):
        return (
            -1 * np.sign(self.close - self.close.shift(7 * self.shiftLag))
        ) + self.close.diff(7 * self.diffLag) * (
            1
            + (1 + self.close.pct_change().rolling(250 * self.rollingLag).sum()).rank(
                pct=True
            )
        )

    def alpha20(self):
        return (
            -1
            * (self.open - self.high.shift(1 * self.shiftLag)).rank(pct=True)
            * (self.open - self.close.shift(1 * self.shiftLag)).rank(pct=True)
            * (self.open - self.low.shift(1 * self.shiftLag)).rank(pct=True)
        )

    def alpha21(self):
        c1 = (
            self.close.rolling(8 * self.rollingLag).mean()
            + self.close.rolling(8 * self.rollingLag).std()
        ) < self.close.rolling(2 * self.rollingLag).mean()
        c2 = self.close.rolling(2 * self.rollingLag).mean() < (
            self.close.rolling(8 * self.rollingLag).mean()
            - self.close.rolling(8 * self.rollingLag).std()
        )
        c3 = (1 < (self.vol / self.adv(20))) | ((self.vol / self.adv(20)) == 1)
        return np.where(c1, -1, np.where(c2, 1, np.where(c3, 1, -1)))

    def alpha22(self):
        return (
            -1
            * self.high.rolling(5 * self.rollingLag)
            .corr(self.vol)
            .diff(5 * self.diffLag)
            * self.close.rolling(20).std().rank(pct=True)
        )

    def alpha23(self):
        return np.where(
            self.high.rolling(20 * self.rollingLag).mean() < self.high,
            -1 * self.high.diff(2 * self.diffLag),
            0,
        )

    def alpha24(self):
        c = (
            self.close.rolling(100 * self.rollingLag).mean().diff(100 * self.diffLag)
            / self.close.shift(100 * self.shiftLag)
        ) <= 0.05
        return np.where(
            c,
            -1 * (self.close - self.close.rolling(100 * self.rollingLag).min()),
            -1 * self.close.diff(3 * self.diffLag),
        )

    def alpha25(self):
        return (
            -1 * self.ret * self.adv(20) * self.vwap * (self.high - self.close)
        ).rank(pct=True)

    def alpha26(self):
        return -1 * (
            self.rollingValues(self.vol, 5 * self.rollingLag, self.pctRank)
            .rolling(5 * self.rollingLag)
            .corr(self.rollingValues(self.high, 5 * self.rollingLag, self.pctRank))
            .rolling(3 * self.rollingLag)
            .max()
        )

    def alpha27(self):
        return np.where(
            0.5
            < self.vol.rank(pct=True)
            .rolling(6 * self.rollingLag)
            .corr(self.vwap.rank(pct=True))
            .rolling(2 * self.rollingLag)
            .mean(),
            -1,
            1,
        )

    def alpha28(self):
        f = (
            self.adv(20).rolling(5 * self.rollingLag).corr(self.low)
            + (self.high + self.low) / 2
            - self.close
        )
        return self.normalize(np.nan_to_num(f, posinf=0, neginf=0))

    def alpha29(self):
        return (
            self.normalize(
                np.log(
                    (-1 * (self.close - 1).diff(5 * self.diffLag).rank(pct=True))
                    .rank(pct=True)
                    .rank(pct=True)
                    .rolling(2 * self.rollingLag)
                    .min()
                    .cumsum()
                )
            )
            .rank(pct=True)
            .rank(pct=True)
            .cumprod()
            .rolling(5 * self.rollingLag)
            .min()
        ) + self.rollingValues(
            -1 * self.ret.shift(6 * self.shiftLag), 5 * self.rollingLag, np.prod
        )

    def alpha30(self):
        return (
            (
                1
                - (
                    np.sign(self.close - self.close.shift(1 * self.shiftLag))
                    + np.sign(
                        self.close.shift(1 * self.shiftLag)
                        - self.close.shift(2 * self.shiftLag)
                    )
                    + np.sign(
                        self.close.shift(2 * self.shiftLag)
                        - self.close.shift(3 * self.shiftLag)
                    )
                ).rank(pct=True)
            )
            * self.adv(5)
            / self.adv(20)
        )

    def alpha31(self):
        return (
            self.rollingValues(
                (-1 * self.close.diff(10 * self.diffLag).rank(pct=True).rank(pct=True)),
                10 * self.rollingLag,
                self.linearDecay,
            )
            .rank(pct=True)
            .rank(pct=True)
            .rank(pct=True)
            + (-1 * self.close.diff(3 * self.diffLag)).rank(pct=True)
            + np.sign(
                self.normalize(
                    self.adv(20).rolling(12 * self.rollingLag).corr(self.low)
                )
            )
        )

    def alpha32(self):
        return self.normalize(
            self.close.rolling(7 * self.rollingLag).mean() - self.close
        ) + (
            20
            * self.otherLag
            * self.normalize(
                self.vwap.rolling(230 * self.rollingLag).corr(
                    self.close.shift(5 * self.shiftLag)
                )
            )
        )

    def alpha33(self):
        return (-1 * (1 - self.open / self.close)).rank(pct=True)

    def alpha34(self):
        return (
            1
            - (
                self.ret.rolling(2 * self.rollingLag).std()
                / self.ret.rolling(5 * self.rollingLag).std()
            ).rank(pct=True)
            + (1 - self.close.diff(self.diffLag).rank(pct=True))
        ).rank(pct=True)

    def alpha35(self):
        return (
            self.rollingValues(self.vol, 32 * self.rollingLag, self.pctRank)
            * (
                1
                - self.rollingValues(
                    self.close + self.high - self.low,
                    16 * self.rollingLag,
                    self.pctRank,
                )
            )
            * (1 - self.rollingValues(self.ret, 32 * self.rollingLag, self.pctRank))
        )

    def alpha36(self):
        return (
            2.21
            * (self.close - self.open)
            .rolling(15 * self.rollingLag)
            .corr(self.vol.shift(self.shiftLag))
            .rank(pct=True)
            + 0.7 * (self.open - self.close).rank(pct=True)
            + 0.73
            * self.rollingValues(
                (-1 * self.ret).shift(6 * self.shiftLag),
                5 * self.rollingLag,
                self.pctRank,
            ).rank(pct=True)
            + self.vwap.rolling(6 * self.rollingLag)
            .corr(self.adv(20))
            .abs()
            .rank(pct=True)
            + 0.6
            * (
                (self.close.rolling(200 * self.rollingLag).mean() - self.open)
                * (self.close - self.open)
            ).rank(pct=True)
        )

    def alpha37(self):
        return (self.open - self.close).shift(self.shiftLag).rolling(
            200 * self.rollingLag
        ).corr(self.close).rank(pct=True) * (self.open - self.close).rank(pct=True)

    def alpha38(self):
        return (
            -1
            * self.rollingValues(self.close, 10 * self.rollingLag, self.pctRank).rank(
                pct=True
            )
            * (self.close / self.open).rank(pct=True)
        )

    def alpha39(self):
        return (
            -1
            * self.close.diff(7 * self.diffLag).rank(pct=True)
            * (
                1
                - self.rollingValues(
                    self.vol / self.adv(20), 9 * self.rollingLag, self.linearDecay
                ).rank(pct=True)
            )
            * (1 + self.ret.rolling(250 * self.rollingLag).sum().rank(pct=True))
        )

    def alpha40(self):
        return (
            -1
            * self.high.rolling(10 * self.rollingLag).std().rank(pct=True)
            * self.high.rolling(10 * self.rollingLag).corr(self.vol)
        )

    def alpha41(self):
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha42(self):
        return (self.vwap - self.close).rank(pct=True) / (self.vwap + self.close).rank(
            pct=True
        )

    def alpha43(self):
        return self.rollingValues(
            self.vol / self.adv(20), 20 * self.rollingLag, self.pctRank
        ) * self.rollingValues(
            -1 * self.close.diff(7 * self.diffLag), 8 * self.rollingLag, self.pctRank
        )

    def alpha44(self):
        return -1 * self.high.rolling(5 * self.rollingLag).corr(self.vol.rank(pct=True))

    def alpha45(self):
        return (
            -1
            * self.close.shift(5 * self.shiftLag)
            .rolling(20 * self.rollingLag)
            .mean()
            .rank(pct=True)
            * self.close.rolling(2 * self.rollingLag).corr(self.vol)
            * self.close.rolling(5 * self.rollingLag)
            .sum()
            .rolling(2 * self.rollingLag)
            .corr(self.close.rolling(20 * self.rollingLag).sum())
            .rank(pct=True)
        )

    def alpha46(self):
        c1 = 0.25 < (
            (
                self.close.shift(20 * self.shiftLag)
                - self.close.shift(10 * self.shiftLag)
            )
            / (10 * self.shiftLag)
            - (self.close.shift(10 * self.shiftLag) - self.close) / (10 * self.shiftLag)
        )
        c2 = (
            (
                self.close.shift(20 * self.shiftLag)
                - self.close.shift(10 * self.shiftLag)
            )
            / (10 * self.shiftLag)
            - (self.close.shift(10 * self.shiftLag) - self.close) / (10 * self.shiftLag)
        ) < 0
        return np.where(
            c1,
            -1,
            np.where(c2, 1, -1 * (self.close - self.close.shift(1 * self.shiftLag))),
        )

    def alpha47(self):
        return (1 / self.close).rank(pct=True) * self.vol / self.adv(20) * self.high * (
            self.high - self.close
        ).rank(pct=True) / self.high.rolling(5 * self.rollingLag).mean() - (
            self.vwap - self.vwap.shift(5 * self.rollingLag)
        ).rank(
            pct=True
        )

    # def alpha48(self):
    # (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    # return

    def alpha49(self):
        return np.where(
            (
                self.close.shift(20 * self.shiftLag)
                - self.close.shift(10 * self.shiftLag)
            )
            / (10 * self.shiftLag)
            - (self.close.shift(10 * self.shiftLag) - self.close) / (10 * self.shiftLag)
            < -0.1,
            1,
            -1 * (self.close - self.close.shift(1 * self.shiftLag)),
        )

    def alpha50(self):
        return (
            -1
            * self.vol.rank(pct=True)
            .rolling(5 * self.rollingLag)
            .corr(self.vwap.rank(pct=True))
            .rank(pct=True)
            .rolling(5 + self.rollingLag)
            .max()
        )

    def alpha51(self):
        return np.where(
            (
                self.close.shift(20 * self.shiftLag)
                - self.close.shift(10 * self.shiftLag)
            )
            / (10 * self.shiftLag)
            - (self.close.shift(10 * self.shiftLag) - self.close) / (10 * self.shiftLag)
            < -0.05,
            1,
            -1 * (self.close - self.close.shift(1 * self.shiftLag)),
        )

    def alpha52(self):
        return (
            (
                -1 * self.low.rolling(5 * self.rollingLag).min()
                + self.low.rolling(5 * self.rollingLag).min().shift(5 * self.shiftLag)
            )
            * (
                (
                    self.ret.rolling(240 * self.rollingLag).sum()
                    - self.ret.rolling(20 * self.rollingLag).sum()
                )
                / (240 * self.rollingLag)
            ).rank(pct=True)
        ) * self.rollingValues(self.vol, 5 * self.rollingLag, self.pctRank)

    def alpha53(self):
        return -1 * (
            ((self.close - self.low) - (self.high - self.close))
            / (self.close - self.low)
        ).diff(9 * self.diffLag)

    def alpha54(self):
        return (-1 * (self.low - self.close) * self.open ** 5) / (
            (self.low - self.high) * self.close ** 5
        )

    def alpha55(self):
        return -1 * (
            self.close
            - self.low.rolling(12 * self.rollingLag).min()
            / self.high.rolling(12 * self.rollingLag).max()
            - self.low.rolling(12 * self.rollingLag).min()
        ).rank(pct=True).rolling(6 * self.rollingLag).corr(self.vol.rank(pct=True))

    # def alpha56(self):
    #  (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # return

    def alpha57(self):
        return 0 - (self.close - self.vwap) / self.rollingValues(
            self.rollingValues(self.close, 30 * self.rollingLag, np.argmax).rank(
                pct=True
            ),
            2 * self.rollingLag,
            self.linearDecay,
        )

    # def alpha58(self):
    #     return  (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))

    # def alpha59(self):
    # return (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))

    def alpha60(self):
        return 0 - (
            2
            * self.normalize(
                (
                    (
                        ((self.close - self.low) - (self.high - self.close))
                        / (self.high - self.low)
                    )
                    * self.vol
                ).rank(pct=True)
            )
            - self.normalize(
                self.rollingValues(self.close, 10 * self.rollingLag, np.argmax).rank(
                    pct=True
                )
            )
        )

    def alpha61(self):
        return (self.vwap - self.vwap.rolling(16 * self.rollingLag).min()).rank(
            pct=True
        ) < self.vwap.rolling(18 * self.rollingLag).corr(self.adv(180)).rank(pct=True)

    def alpha62(self):
        return -1 * (
            self.vwap.rolling(10 * self.rollingLag)
            .corr(self.adv(20).rolling(22 * self.rollingLag).sum())
            .rank(pct=True)
            < (
                self.open.rank(pct=True) * 2
                < (
                    ((self.high + self.low) / 2).rank(pct=True)
                    + self.high.rank(pct=True)
                )
            )
        )

    # def alpha63(self):
    #     return ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)

    def alpha64(self, m=0.178404):
        return -1 * (
            (self.open * m + self.low * (1 - m))
            .rolling(13 * self.rollingLag)
            .sum()
            .rolling(17 * self.rollingLag)
            .corr(self.adv(120).rolling(13 * self.rollingLag).sum())
            .rank(pct=True)
            < ((self.high + self.low) / 2 * m + self.vwap * (1 - m))
            .diff(4 * self.diffLag)
            .rank(pct=True)
        )

    def alpha65(self, m=0.00817205):
        return -1 * (
            (self.open * m + self.vwap * (1 - m))
            .rolling(6 * self.rollingLag)
            .corr(self.adv(60).rolling(9 * self.rollingLag).sum())
            .rank(pct=True)
            < (self.open - self.open.rolling(14 * self.rollingLag).min()).rank(pct=True)
        )

    def alpha66(self):
        return -1 * (
            self.rollingValues(
                self.vwap.diff(4 * self.diffLag), 7 * self.rollingLag, self.linearDecay
            ).rank(pct=True)
            + self.rollingValues(
                self.rollingValues(
                    ((self.low * 0.96633) + (self.low * (1 - 0.96633)) - self.low)
                    / (self.open - (self.high + self.low) / 2),
                    11 * self.rollingLag,
                    self.linearDecay,
                ),
                7 * self.rollingLag,
                self.pctRank,
            )
        )

    # def alpha67(self):
    #     return  ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)

    def alpha68(self):
        return -1 * (
            self.rollingValues(
                self.high.rank(pct=True)
                .rolling(9 * self.rollingLag)
                .corr(self.adv(15).rank(pct=True)),
                14 * self.rollingLag,
                self.pctRank,
            )
            < ((self.close * 0.518371) + (self.low * (1 - 0.518371)))
            .diff(1 * self.diffLag)
            .rank(pct=True)
        )

    # def alpha69(self):
    #     return  ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)

    # def alpha70(self):
    #     return ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

    def alpha71(self):
        d = self.rollingValues(
            self.rollingValues(
                ((self.low + self.open) - (self.vwap * 2)).rank(pct=True) ** 2,
                16 * self.rollingLag,
                self.linearDecay,
            ),
            4 * self.rollingLag,
            self.pctRank,
        )
        return np.maximum(
            self.rollingValues(
                self.rollingValues(
                    self.rollingValues(self.close, 3 * self.rollingLag, self.pctRank)
                    .rolling(18 * self.rollingLag)
                    .corr(
                        self.rollingValues(
                            self.adv(180), 12 * self.rollingLag, self.pctRank
                        )
                    ),
                    4 * self.rollingLag,
                    self.linearDecay,
                ),
                16 * self.rollingLag,
                self.pctRank,
            ),
            d,
        )

    def alpha72(self):
        return (
            self.rollingValues(
                ((self.high + self.low) / 2)
                .rolling(9 * self.rollingLag)
                .corr(self.adv(40)),
                10 * self.rollingLag,
                self.linearDecay,
            ).rank(pct=True)
            / self.rollingValues(
                self.rollingValues(self.vwap, 4 * self.rollingLag, self.pctRank)
                .rolling(7 * self.rollingLag)
                .corr(self.rollingValues(self.vol, 19, self.pctRank)),
                3 * self.rollingLag,
                self.linearDecay,
            ).rank(pct=True)
        )

    def alpha73(self):
        return np.maximum(
            self.rollingValues(
                self.vwap.diff(5 * self.diffLag), 3 * self.rollingLag, self.linearDecay
            ).rank(pct=True),
            self.rollingValues(
                self.rollingValues(
                    -1
                    * ((self.open * 0.147155) + (self.low * (1 - 0.147155))).diff(
                        2 * self.diffLag
                    )
                    / ((self.open * 0.147155) + (self.low * (1 - 0.147155))),
                    3 * self.rollingLag,
                    self.linearDecay,
                ),
                17 * self.rollingLag,
                self.pctRank,
            ),
        )

    def alpha74(self):
        return -1 * (
            self.close.rolling(15 * self.rollingLag)
            .corr(self.adv(30).rolling(37 * self.rollingLag).sum())
            .rank(pct=True)
            < ((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))
            .rank(pct=True)
            .rolling(11 * self.rollingLag)
            .corr(self.vol.rank(pct=True))
            .rank(pct=True)
        )

    def alpha75(self):
        return (
            self.vwap.rolling(4 * self.rollingLag).corr(self.vol).rank(pct=True)
            < self.low.rank(pct=True)
            .rolling(12 * self.rollingLag)
            .corr(self.adv(50).rank(pct=True))
        ) * 1

    # def alpha76(self):
    #     (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
    #     return np.maximum(self.rollingValues(self.vwap.diff(1),12,self.linearDecay).rank(pct=True),)

    def alpha77(self):
        return np.minimum(
            self.rollingValues(
                ((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)),
                20 * self.rollingLag,
                self.linearDecay,
            ).rank(pct=True),
            self.rollingValues(
                ((self.high + self.low) / 2)
                .rolling(3 * self.rollingLag)
                .corr(self.adv(40)),
                6 * self.rollingLag,
                self.linearDecay,
            ).rank(pct=True),
        )

    def alpha78(self):
        return ((self.low * 0.352233) + (self.vwap * (1 - 0.352233))).rolling(
            round(19.7428 * self.rollingLag)
        ).sum().rolling(round(6.83313 * self.rollingLag)).corr(
            self.adv(40).rolling(round(19.7428 * self.rollingLag)).sum()
        ).rank(
            pct=True
        ) ** (
            self.vwap.rank(pct=True)
            .rolling(round(5.77492 * self.rollingLag))
            .corr(self.vol.rank(pct=True))
            .rank(pct=True)
        )

    # def alpha79(self):
    #     return  (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))

    # def alpha80(self):
    #     return  ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

    def alpha81(self):
        return -1 * (
            pd.Series(
                np.log(
                    self.rollingValues(
                        self.vwap.rolling(round(8.47743 * self.rollingLag))
                        .corr(
                            self.adv(10).rolling(round(49.6054 * self.rollingLag)).sum()
                        )
                        .rank(pct=True)
                        ** 4,
                        round(14.9655 * self.rollingLag),
                        np.prod,
                    )
                ),
                index=self.vwap.index,
            ).rank(pct=True)
            < self.vwap.rank(pct=True)
            .rolling(round(5.07914 * self.rollingLag))
            .corr(self.vol.rank(pct=True))
            .rank(pct=True)
        )

    # def alpha82(self):
    #     return (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

    def alpha83(self):
        m = (self.high - self.low) / self.close.rolling(5 * self.rollingLag).mean()
        return (
            m.shift(2 * self.shiftLag).rank(pct=True)
            * self.vol.rank(pct=True).rank(pct=True)
            / (m / (self.vwap - self.close))
        )

    def alpha84(self):
        return (
            self.rollingValues(
                self.vwap.rolling(round(15.3217 * self.rollingLag)).max(),
                round(20.7127 * self.rollingLag),
                self.pctRank,
            )
            ** self.close.diff(round(4.96796 * self.diffLag))
        )

    def alpha85(self):
        return ((self.high * 0.876703) + (self.close * (1 - 0.876703))).rolling(
            round(9.61331 * self.rollingLag)
        ).corr(self.adv(30)) ** (
            self.rollingValues(
                ((self.high + self.low) / 2),
                round(3.70596 * self.rollingLag),
                self.pctRank,
            )
            .rolling(round(7.11408 * self.rollingLag))
            .corr(
                self.rollingValues(
                    self.vol, round(10.1595 * self.rollingLag), self.pctRank
                )
            )
            .rank(pct=True)
        )

    def alpha86(self):
        return -1 * (
            self.rollingValues(
                self.close.rolling(round(6.00049 * self.rollingLag)).corr(
                    self.adv(20).rolling(round(14.7444 * self.rollingLag)).sum()
                ),
                round(20.4195 * self.rollingLag),
                self.pctRank,
            )
            < ((self.open + self.close) - (self.vwap + self.open)).rank(pct=True)
        )

    # def alpha87(self):
    #     return (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)

    def alpha88(self):
        return np.minimum(
            self.rollingValues(
                (
                    self.open.rank(pct=True)
                    + self.low.rank(pct=True)
                    - (self.high.rank(pct=True) + self.close.rank(pct=True))
                ),
                round(8.06882 * self.rollingLag),
                self.linearDecay,
            ).rank(pct=True),
            self.rollingValues(
                self.rollingValues(
                    self.rollingValues(
                        self.close, round(8.44728 * self.rollingLag), self.pctRank
                    )
                    .rolling(round(8.01266 * self.rollingLag))
                    .corr(
                        self.rollingValues(
                            self.adv(60), round(20.6966 * self.rollingLag), self.pctRank
                        )
                    ),
                    round(6.65053 * self.rollingLag),
                    self.linearDecay,
                ),
                round(2.61957 * self.rollingLag),
                self.pctRank,
            ),
        )

    # def alpha89(self):
    #     return (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))

    # def alpha90(self):
    #     return ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)

    # def alpha91(self):
    #     return ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)

    def alpha92(self):
        return np.minimum(
            self.rollingValues(
                self.rollingValues(
                    (((self.high + self.low) / 2) + self.close)
                    < (self.low + self.open) * 1,
                    round(14.7221 * self.rollingLag),
                    self.linearDecay,
                ),
                round(18.8683 * self.rollingLag),
                self.pctRank,
            ),
            self.rollingValues(
                self.rollingValues(
                    self.low.rank(pct=True)
                    .rolling(round(7.58555 * self.rollingLag))
                    .corr(self.adv(30).rank(pct=True)),
                    round(6.94024 * self.rollingLag),
                    self.linearDecay,
                ),
                round(6.80584 * self.rollingLag),
                self.pctRank,
            ),
        )

    # def alpha93(self):
    #     return  (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))

    def alpha94(self):
        return -1 * (
            (
                self.vwap - self.vwap.rolling(round(11.5783 * self.rollingLag)).min()
            ).rank(pct=True)
            ** (
                self.rollingValues(
                    self.rollingValues(
                        self.vwap, round(19.6462 * self.rollingLag), self.pctRank
                    )
                    .rolling(round(18.0926 * self.rollingLag))
                    .corr(
                        self.rollingValues(
                            self.adv(60), round(4.02992 * self.rollingLag), self.pctRank
                        )
                    ),
                    round(2.70756 * self.rollingLag),
                    self.pctRank,
                )
            )
        )

    def alpha95(self):
        return (
            (
                self.open - self.open.rolling(round(12.4105 * self.rollingLag)).min()
            ).rank(pct=True)
            < self.rollingValues(
                ((self.high + self.low) / 2)
                .rolling(round(19.1351 * self.rollingLag))
                .sum()
                .rolling(round(12.8742 * self.rollingLag))
                .corr(self.adv(40).rolling(round(19.1351 * self.rollingLag)).sum())
                .rank(pct=True)
                ** 5,
                round(11.7584 * self.rollingLag),
                self.pctRank,
            )
        ) * 1

    def alpha96(self):
        arr1 = self.rollingValues(
            self.rollingValues(
                self.vwap.rank(pct=True)
                .rolling(round(3.83878 * self.rollingLag))
                .corr(self.vol.rank(pct=True)),
                round(4.16783 * self.rollingLag),
                self.linearDecay,
            ),
            round(8.38151 * self.rollingLag),
            self.pctRank,
        )
        arr2 = self.rollingValues(
            self.rollingValues(
                self.rollingValues(
                    self.rollingValues(
                        self.close, round(7.45404 * self.rollingLag), self.pctRank
                    )
                    .rolling(round(3.65459 * self.rollingLag))
                    .corr(
                        self.rollingValues(
                            self.adv(60), round(4.13242 * self.rollingLag), self.pctRank
                        )
                    ),
                    round(12.6556 * self.rollingLag),
                    np.argmax,
                ),
                round(14.0365 * self.rollingLag),
                self.linearDecay,
            ),
            round(13.4143 * self.rollingLag),
            self.pctRank,
        )
        return -1 * np.maximum(arr1, arr2)

    # def alpha97(self):
    #     return ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

    def alpha98(self):
        return (
            self.rollingValues(
                self.vwap.rolling(round(4.58418 * self.rollingLag)).corr(
                    self.adv(5).rolling(round(26.4719 * self.rollingLag)).sum()
                ),
                round(7.18088 * self.rollingLag),
                self.linearDecay,
            ).rank(pct=True)
            - self.rollingValues(
                self.rollingValues(
                    self.rollingValues(
                        self.open.rank(pct=True)
                        .rolling(round(20.8187 * self.rollingLag))
                        .corr(self.adv(15).rank(pct=True)),
                        round(8.62571 * self.rollingLag),
                        np.argmax,
                    ),
                    round(6.95668 * self.rollingLag),
                    self.pctRank,
                ),
                round(8.07206 * self.rollingLag),
                self.linearDecay,
            ).rank(pct=True)
        )

    def alpha99(self):
        return -1 * (
            ((self.high + self.low) / 2)
            .rolling(round(19.8975 * self.rollingLag))
            .sum()
            .rolling(round(8.8136 * self.rollingLag))
            .corr(self.adv(60).rolling(round(19.8975 * self.rollingLag)).sum())
            .rank(pct=True)
            < self.low.rolling(round(6.28259 * self.rollingLag))
            .corr(self.vol)
            .rank(pct=True)
        )

    # def alpha100(self):
    #     return  (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))

    def alpha101(self):
        return (self.close - self.open) / (
            (self.high - self.low) + 0.001 * self.otherLag
        )

    @staticmethod
    def getAll(
        open,
        high,
        low,
        close,
        vol,
        aCols=[f"alpha{i}" for i in range(1, 102, 1)],
        rollingLag=1,
        diffLag=1,
        shiftLag=1,
        otherLag=1,
    ):
        alphas = FormulaicAlphas101(
            open,
            high,
            low,
            close,
            vol,
            rollingLag=rollingLag,
            diffLag=diffLag,
            shiftLag=shiftLag,
            otherLag=otherLag,
        )
        out = pd.DataFrame(index=close.index)
        t = time.time()
        for i, c in enumerate(aCols):
            try:
                out[c] = getattr(alphas, c)()
            except (AttributeError) as e:
                pass  # print(e)
            report_progress(i + 1, len(aCols), t, sys._getframe().f_code.co_name)
        return out


def getTmp(df_path = "data/df.feather"):
    df = pd.read_feather(df_path).set_index("date")
    idx = pd.IndexSlice
    return df.loc[idx["2020"]]


def getAllAlphas(tmp, alphas, plotCorr=True):
    out = pd.DataFrame(index=tmp.index)
    aCols = [f"alpha{i}" for i in range(1, 102, 1)]
    t = time.time()
    for i, c in enumerate(aCols):
        try:
            out[c] = getattr(alphas, c)()
        except (AttributeError):
            pass
        report_progress(i + 1, len(aCols), t, sys._getframe().f_code.co_name)
    if plotCorr:
        sns.heatmap(out.corr(), linewidth=0.5, cmap="PiYG")
        plt.show()
    return out


if __name__ == "__main__":
    path = "data/df.feather"
    tmp = getTmp(path)
    alphas = FormulaicAlphas101(tmp["o"],tmp["h"],tmp["l"],tmp["c"],tmp["vol"])
    getAllAlphas(tmp,alphas)
