"""Binomial option pricing model.

Implements a Cox-Ross-Rubinstein style recombining binomial tree that can
price European and American calls or puts.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import List


@dataclass
class OptionSpec:
    """Inputs required to price an option with the binomial tree model."""

    spot: float
    strike: float
    maturity: float  # in years
    rate: float  # continuously compounded risk-free rate
    volatility: float  # annualized volatility
    steps: int
    is_call: bool
    american: bool = False
    dividend_yield: float = 0.0  # continuous dividend yield

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("Spot and strike must be positive.")


def _risk_neutral_probability(u: float, d: float, r: float, q: float, dt: float) -> float:
    growth = exp((r - q) * dt)
    probability = (growth - d) / (u - d)
    if not 0 <= probability <= 1:
        raise ValueError("Risk-neutral probability out of bounds; adjust parameters.")
    return probability


def _terminal_asset_prices(spec: OptionSpec, u: float, d: float) -> List[float]:
    return [spec.spot * (u ** j) * (d ** (spec.steps - j)) for j in range(spec.steps + 1)]


def _option_payoff(price: float, strike: float, is_call: bool) -> float:
    return max(price - strike, 0.0) if is_call else max(strike - price, 0.0)


def binomial_option_price(spec: OptionSpec) -> float:
    """Price an option using a recombining binomial tree.

    Parameters
    ----------
    spec: OptionSpec
        Input parameters describing the option contract and market data.

    Returns
    -------
    float
        The option price at the initial node.
    """

    dt = spec.maturity / spec.steps
    u = exp(spec.volatility * sqrt(dt))
    d = 1 / u
    p = _risk_neutral_probability(u, d, spec.rate, spec.dividend_yield, dt)
    discount = exp(-spec.rate * dt)

    asset_prices = _terminal_asset_prices(spec, u, d)
    option_values = [_option_payoff(price, spec.strike, spec.is_call) for price in asset_prices]

    for step in range(spec.steps - 1, -1, -1):
        next_values = []
        for i in range(step + 1):
            continuation = discount * (p * option_values[i + 1] + (1 - p) * option_values[i])
            if spec.american:
                underlying = spec.spot * (u ** i) * (d ** (step - i))
                exercise = _option_payoff(underlying, spec.strike, spec.is_call)
                next_values.append(max(exercise, continuation))
            else:
                next_values.append(continuation)
        option_values = next_values
    return option_values[0]


if __name__ == "__main__":
    example = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=100,
        is_call=True,
        american=False,
    )
    price = binomial_option_price(example)
    print(f"European call price: {price:.4f}")
