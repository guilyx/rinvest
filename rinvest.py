import pandas as pd
import matplotlib.pyplot as plt
from yfinance import Ticker
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime
import json

plt.style.use("ggplot")


@dataclass
class StrategyPerformance:
    name: str
    total_return: float
    daily_portfolio_values: pd.Series
    pnl: pd.Series
    investments: pd.Series


@dataclass
class RecurrentInvestment:
    amount: float
    frequency: str
    start_date: Union[str, datetime]
    end_date: Union[str, datetime]


class Strategy:
    def __init__(
        self,
        name: str,
        initial_investment: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        recurrent_investment: Optional[RecurrentInvestment] = None,
    ):
        self.name = name
        self.initial_investment = initial_investment
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.recurrent_investment = recurrent_investment


class InvestmentStrategy(Strategy):
    def __init__(
        self,
        asset: str,
        initial_investment: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        recurrent_investment: Optional[RecurrentInvestment] = None,
        annual_interest_rate: Optional[float] = None,
    ):
        super().__init__(
            asset, initial_investment, start_date, end_date, recurrent_investment
        )
        self.asset = asset
        self.annual_interest_rate = annual_interest_rate

    def calculate_return(self) -> StrategyPerformance:
        data = Ticker(self.asset).history(start=self.start_date, end=self.end_date)[
            "Close"
        ]
        data.index = data.index.tz_convert(None)  # make index tz-naive
        initial_price = data[0]
        daily_interest_rate = (
            (1 + self.annual_interest_rate) ** (1 / 365) - 1
            if self.annual_interest_rate is not None
            else 0
        )
        daily_portfolio_values = (data / initial_price) * self.initial_investment
        total_recurrent_investment = 0
        recurrent_investments = pd.Series(0, index=daily_portfolio_values.index)
        if self.recurrent_investment:
            recurrent_initial_investment = self.recurrent_investment.amount
            recurrent_investment_frequency = self.recurrent_investment.frequency
            recurrent_investment_dates = pd.date_range(
                start=data.index[0],
                end=data.index[-1],
                freq=recurrent_investment_frequency,
            )

            for date in recurrent_investment_dates:
                recurrent_investments.loc[date] = recurrent_initial_investment

            total_recurrent_investment = recurrent_investments.sum()
            daily_portfolio_values += recurrent_investments.cumsum()

        if self.annual_interest_rate:
            daily_portfolio_values *= (1 + daily_interest_rate) ** np.arange(
                len(daily_portfolio_values)
            )

        pnl = (
            daily_portfolio_values - daily_portfolio_values.shift(1)
        ) - recurrent_investments
        total_return = (
            daily_portfolio_values[-1]
            - self.initial_investment
            - total_recurrent_investment
        )
        return StrategyPerformance(
            self.name, total_return, daily_portfolio_values, pnl, recurrent_investments
        )


class SavingsStrategy(Strategy):
    def __init__(
        self,
        name: str,
        annual_interest_rate: float,
        initial_investment: float,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        recurrent_investment: Optional[RecurrentInvestment] = None,
    ):
        super().__init__(
            name, initial_investment, start_date, end_date, recurrent_investment
        )
        self.annual_interest_rate = annual_interest_rate

    def calculate_return(self) -> StrategyPerformance:
        daily_interest_rate = (1 + self.annual_interest_rate) ** (1 / 365) - 1
        index = pd.date_range(start=self.start_date, end=self.end_date)
        daily_portfolio_values = pd.Series(
            self.initial_investment
            * (1 + daily_interest_rate) ** np.arange(len(index)),
            index=index,
        )
        total_recurrent_investment = 0
        recurrent_investments = pd.Series(0, index=daily_portfolio_values.index)

        if self.recurrent_investment:
            recurrent_initial_investment = self.recurrent_investment.amount
            recurrent_investment_frequency = self.recurrent_investment.frequency
            recurrent_investment_dates = pd.date_range(
                start=index[0], end=index[-1], freq=recurrent_investment_frequency
            )

            for date in recurrent_investment_dates:
                recurrent_investments.loc[date] = recurrent_initial_investment

            total_recurrent_investment = recurrent_investments.sum()
            daily_portfolio_values += recurrent_investments.cumsum() * (
                1 + daily_interest_rate
            ) ** np.arange(len(recurrent_investments))

        pnl = (
            daily_portfolio_values - daily_portfolio_values.shift(1)
        ) - recurrent_investments
        total_return = (
            daily_portfolio_values[-1]
            - self.initial_investment
            - total_recurrent_investment
        )
        return StrategyPerformance(
            self.name, total_return, daily_portfolio_values, pnl, recurrent_investments
        )


class BackTester:
    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies

    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy)

    def backtest(self) -> Dict[str, StrategyPerformance]:
        try:
            performances = {}
            for strategy in self.strategies:
                performances[strategy.name] = strategy.calculate_return()
            return performances
        except:
            print("Data was not available!")
            return {}

    def plot_portfolio_values(
        self, performances: Dict[str, StrategyPerformance], frequencies: List[str]
    ):
        for f in frequencies:
            is_valid_frequency(f)
        if not performances:
            return

        n_frequencies = len(frequencies)
        fig, axs = plt.subplots(
            n_frequencies, 1, figsize=(14, 7 * n_frequencies), squeeze=False
        )
        titles = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly"}

        for i, ax in enumerate(axs.flatten()):
            for name, performance in performances.items():
                data = performance.daily_portfolio_values.resample(
                    frequencies[i]
                ).last()
                ax.plot(data, label=name)

            total_daily_values = (
                pd.concat(
                    [p.daily_portfolio_values for p in performances.values()], axis=1
                )
                .fillna(method="ffill")
                .sum(axis=1)
            )

            total_daily_investments = (
                pd.concat([p.investments for p in performances.values()], axis=1)
                .fillna(method="ffill")
                .sum(axis=1)
            )

            total_daily_values = total_daily_values.resample(frequencies[i]).last()
            total_daily_investments = total_daily_investments.cumsum().resample(
                frequencies[i]
            ).last()

            ax.plot(
                total_daily_values,
                label="Cumulative Value",
                color="green",
                linestyle="--",
            )

            ax.plot(
                total_daily_investments,
                label="Cumulative Investment",
                color="red",
                linestyle="--",
            )

            ax.set_title(f"Portfolio Value ({titles[frequencies[i]]})")
            ax.legend()

    def plot_pnl(self, performances: Dict[str, StrategyPerformance], frequency="D"):
        is_valid_frequency(frequency)
        if not performances:
            return

        titles = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly"}
        pnl_df = pd.DataFrame()

        for name, performance in performances.items():
            data = performance.pnl.resample(frequency).sum()
            pnl_df[name] = data

        pnl_df.plot.bar(stacked=True, figsize=(14, 7))
        plt.title(f"{titles[frequency]} PNL")

    def plot_performance(
        self, performances: Dict[str, StrategyPerformance], frequency="D"
    ):
        is_valid_frequency(frequency)
        if not performances:
            return

        titles = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly"}
        performance_df = pd.DataFrame()

        for name, performance in performances.items():
            total_investments = performance.daily_portfolio_values
            performance_pct_change = (
                (performance.pnl / total_investments).resample(frequency).sum()
            )
            performance_df[name] = performance_pct_change

        performance_df.plot.bar(stacked=True, figsize=(14, 7))
        plt.title(f"{titles[frequency]} Performance")

    def show_plots(self):
        plt.tight_layout()
        plt.show()


class StrategyComparator:
    def __init__(self, performances: Dict[str, Dict[str, StrategyPerformance]]):
        self.performances = performances

    def plot_balance(self, frequencies: List[str]):
        for f in frequencies:
            is_valid_frequency(f)
        if not frequencies:
            return

        n_frequencies = len(frequencies)
        fig, axs = plt.subplots(
            n_frequencies, 1, figsize=(14, 7 * n_frequencies), squeeze=False
        )
        titles = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly"}

        for i, ax in enumerate(axs.flatten()):
            for name, performance in self.performances.items():
                total_daily_values = (
                    pd.concat(
                        [p.daily_portfolio_values for p in performance.values()], axis=1
                    )
                    .fillna(method="ffill")
                    .sum(axis=1)
                )
                total_daily_values = total_daily_values.resample(frequencies[i]).last()
                ax.plot(total_daily_values, label=f"Cumulative [{name}]")

                ax.set_title(f"Portfolio Value ({titles[frequencies[i]]})")
                ax.legend()

    def show_plots(self):
        plt.show()


def convert_to_json(
    strategies: List[Union[InvestmentStrategy, SavingsStrategy]], filename: str
):
    strategies_json = []

    for strategy in strategies:
        strategy_dict = strategy.__dict__
        strategy_dict["start_date"] = strategy_dict["start_date"].isoformat()
        strategy_dict["end_date"] = strategy_dict["end_date"].isoformat()
        strategy_dict["type"] = strategy.__class__.__name__
        if strategy.recurrent_investment is not None:
            strategy_dict[
                "recurrent_investment"
            ] = strategy.recurrent_investment.__dict__

        strategies_json.append(strategy_dict)

    with open(filename, "w") as f:
        json.dump(strategies_json, f, indent=4)


def load_json(json_file: str):
    basket = {}

    with open(json_file, "r") as f:
        strategies_data = json.load(f)

    for name, strategies in strategies_data.items():
        basket[name] = []
        for data in strategies:
            if data["type"] == "InvestmentStrategy":
                recurrent_investment = RecurrentInvestment(
                    data["recurrent_investment"]["amount"],
                    data["recurrent_investment"]["frequency"],
                    data["recurrent_investment"]["start_date"],
                    data["recurrent_investment"]["end_date"],
                )
                apr = data.get("annual_interest_rate")
                basket[name].append(
                    InvestmentStrategy(
                        data["asset"],
                        data["initial_investment"],
                        data["start_date"],
                        data["end_date"],
                        recurrent_investment,
                        apr,
                    )
                )
            elif data["type"] == "SavingsStrategy":
                recurrent_investment = RecurrentInvestment(
                    data["recurrent_investment"]["amount"],
                    data["recurrent_investment"]["frequency"],
                    data["recurrent_investment"]["start_date"],
                    data["recurrent_investment"]["end_date"],
                )
                basket[name].append(
                    SavingsStrategy(
                        data["name"],
                        data["annual_interest_rate"],
                        data["initial_investment"],
                        data["start_date"],
                        data["end_date"],
                        recurrent_investment,
                    )
                )

    return basket


def is_valid_frequency(frequency: str):
    valid_frequencies = {"D", "W", "M", "Y"}
    if frequency not in valid_frequencies:
        raise ValueError(
            f"Invalid frequency: {frequency}. Allowed values are {valid_frequencies}"
        )


if __name__ == "__main__":
    basket = load_json("strategies.json")
    perfs = {}
    for k, strategies in basket.items():
        tester = BackTester(strategies)
        strategy_performances = tester.backtest()
        perfs[k] = strategy_performances
        tester.plot_portfolio_values(strategy_performances, ["D", "M"])
        tester.plot_pnl(strategy_performances, "M")
        tester.plot_pnl(strategy_performances, "W")
        tester.plot_performance(strategy_performances, "M")
        tester.show_plots()

    comparator = StrategyComparator(perfs)
    comparator.plot_balance(["D", "W", "M"])
    comparator.show_plots()
