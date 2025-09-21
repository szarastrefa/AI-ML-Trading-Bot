"""
Testy strategii tradingowych
"""

import unittest
import pandas as pd
import numpy as np
from app.strategies.fibonacci_strategy import FibonacciTeamStrategy
from app.strategies.smart_money_concept import SmartMoneyConceptStrategy

class TestStrategies(unittest.TestCase):

    def setUp(self):
        """Przygotuj dane testowe"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')

        # Generuj przykładowe dane OHLCV
        prices = np.random.random(100) * 0.001 + 1.0000  # EURUSD-like prices

        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.random(100) * 0.0005,
            'low': prices - np.random.random(100) * 0.0005,
            'close': prices + np.random.random(100) * 0.0002 - 0.0001,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_fibonacci_strategy(self):
        """Test strategii Fibonacci Team"""
        config = {
            "fib_lookback": 20,
            "min_swing_size": 0.002,
            "volume_threshold": 1.5
        }

        strategy = FibonacciTeamStrategy(config)
        signals = strategy.generate_signals(self.test_data)

        self.assertIsInstance(signals, list)
        self.assertEqual(strategy.name, "fibonacci_team")

        # Test informacji o strategii
        info = strategy.get_strategy_info()
        self.assertIn("name", info)
        self.assertIn("features", info)

    def test_smart_money_concept_strategy(self):
        """Test strategii Smart Money Concept"""
        config = {
            "swing_lookback": 20,
            "ob_min_size": 0.001,
            "fvg_min_size": 0.0005
        }

        strategy = SmartMoneyConceptStrategy(config)
        signals = strategy.generate_signals(self.test_data)

        self.assertIsInstance(signals, list)
        self.assertEqual(strategy.name, "smart_money_concept")

        # Test informacji o strategii
        info = strategy.get_strategy_info()
        self.assertIn("name", info)
        self.assertIn("features", info)

    def test_position_sizing(self):
        """Test obliczania wielkości pozycji"""
        config = {}
        strategy = FibonacciTeamStrategy(config)

        account_balance = 10000
        entry_price = 1.0000
        stop_loss = 0.9980  # 20 pips SL

        position_size = strategy.calculate_position_size(account_balance, entry_price, stop_loss)

        self.assertGreater(position_size, 0)
        self.assertLess(position_size, account_balance)  # Nie więcej niż cały kapitał

if __name__ == '__main__':
    unittest.main()
