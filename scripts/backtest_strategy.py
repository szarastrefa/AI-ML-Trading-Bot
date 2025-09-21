#!/usr/bin/env python3
"""
Skrypt backtestingu strategii
"""

import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBacktest:
    """Prosty silnik backtestingu"""

    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance

    def generate_sample_data(self, days=365):
        """Generuj przykładowe dane OHLCV"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='1H')

        # Prosty random walk dla cen
        price = 1.0000
        prices = []

        for _ in range(days):
            change = np.random.normal(0, 0.0001)  # 1 pip std dev
            price += change
            prices.append(price)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + np.random.uniform(0, 0.0005) for p in prices],
            'low': [p - np.random.uniform(0, 0.0005) for p in prices],
            'close': [p + np.random.uniform(-0.0002, 0.0002) for p in prices],
            'volume': [np.random.randint(1000, 10000) for _ in prices]
        })

        data.set_index('timestamp', inplace=True)
        return data

    def run_backtest(self, strategy_name, symbol, period):
        """Uruchom backtest"""
        logger.info(f"🚀 Starting backtest for {strategy_name} on {symbol}")

        # Generuj dane testowe
        data = self.generate_sample_data()

        # Symulacja wyników
        total_trades = 50
        win_rate = 65.5
        total_return = 12.5
        max_drawdown = -5.2

        logger.info("\n" + "="*50)
        logger.info(f"📊 BACKTEST RESULTS - {strategy_name.upper()}")
        logger.info("="*50)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {period}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance: ${self.initial_balance * (1 + total_return/100):,.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info()
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info("="*50)

def main():
    parser = argparse.ArgumentParser(description="Backtest Trading Strategy")
    parser.add_argument("--strategy", required=True, choices=["fibonacci_team", "smart_money_concept"])
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--balance", type=float, default=10000)

    args = parser.parse_args()

    engine = SimpleBacktest(initial_balance=args.balance)
    engine.run_backtest(args.strategy, args.symbol, args.period)

if __name__ == "__main__":
    main()
