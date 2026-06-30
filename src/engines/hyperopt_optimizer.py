"""
Hyperopt 超參數優化器 (Hyperopt Optimizer)

靈感來源：Freqtrade Hyperopt 系統
功能：
1. 使用 Optuna 進行超參數優化
2. 多種損失函數（Sharpe, Sortino, Profit/DrawDown）
3. 並行優化
4. 結果可視化
5. 與 Arena 系統整合

作者：Binance Trading Bot v0.3.0
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pickle

from src.config import settings
from src.logger import logger
from src.engines.backtest_engine import backtest_engine

# 嘗試導入 Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Hyperopt features will be limited.")


class LossFunction(str, Enum):
    """損失函數類型"""
    SHARPE = "sharpe"                     # 夏普比率
    SORTINO = "sortino"                   # 索提諾比率
    PROFIT_DRAWDOWN = "profit_drawdown"   # 利潤/回撤
    CALMAR = "calmar"                     # Calmar 比率
    TOTAL_PROFIT = "total_profit"         # 總利潤


@dataclass
class HyperoptConfig:
    """Hyperopt 配置"""
    # 優化設定
    loss_function: LossFunction = LossFunction.SHARPE
    n_trials: int = 100                   # 優化試驗次數
    timeout: int = 3600                   # 超時時間（秒）
    n_jobs: int = 1                       # 並行任務數

    # 數據設定
    data_days: int = 30                   # 回測數據天數
    starting_balance: float = 10000.0     # 起始資金
    risk_free_rate: float = 0.0           # 無風險利率

    # 回撤設定
    drawdown_mult: float = 0.075          # 回撤懲罰係數（越小懲罰越嚴重）

    # 策略設定
    strategy: str = "momentum"            # 策略名稱
    symbol: str = "BTCUSDT"               # 交易對
    interval: str = "1h"                  # K 線間隔

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "loss_function": self.loss_function.value
        }


@dataclass
class OptimizationResult:
    """優化結果"""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    total_time: float
    loss_function: str
    strategy: str
    created_at: datetime = field(default_factory=datetime.now)
    all_trials: Optional[List[Dict]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": round(self.best_value, 4),
            "n_trials": self.n_trials,
            "total_time": round(self.total_time, 2),
            "loss_function": self.loss_function,
            "strategy": self.strategy,
            "created_at": self.created_at.isoformat(),
            "all_trials": self.all_trials
        }


class SearchSpace:
    """搜索空間定義"""

    @staticmethod
    def momentum_space(trial) -> Dict[str, Any]:
        """動量策略搜索空間"""
        return {
            "rsi_period": trial.suggest_int("rsi_period", 5, 30),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 90),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 10, 40),
            "sma_short": trial.suggest_int("sma_short", 5, 50),
            "sma_long": trial.suggest_int("sma_long", 20, 200),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.001, 0.05),
            "position_ratio": trial.suggest_float("position_ratio", 0.1, 1.0, step=0.1),
        }

    @staticmethod
    def mean_reversion_space(trial) -> Dict[str, Any]:
        """均值回歸策略搜索空間"""
        return {
            "bb_period": trial.suggest_int("bb_period", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.0, step=0.1),
            "rsi_period": trial.suggest_int("rsi_period", 5, 30),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 85),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
            "take_profit": trial.suggest_float("take_profit", 0.01, 0.05, step=0.005),
            "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.05, step=0.005),
        }

    @staticmethod
    def trend_following_space(trial) -> Dict[str, Any]:
        """趨勢跟蹤策略搜索空間"""
        return {
            "ema_short": trial.suggest_int("ema_short", 5, 30),
            "ema_long": trial.suggest_int("ema_long", 20, 100),
            "adx_period": trial.suggest_int("adx_period", 10, 30),
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 35),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 1.0, 4.0, step=0.5),
            "trailing_stop": trial.suggest_float("trailing_stop", 0.01, 0.05, step=0.005),
        }

    @staticmethod
    def ml_model_space(trial) -> Dict[str, Any]:
        """ML 模型超參數搜索空間"""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "prediction_horizon": trial.suggest_int("prediction_horizon", 1, 10),
            "threshold": trial.suggest_float("threshold", 0.005, 0.03, step=0.005),
            "confidence_threshold": trial.suggest_float("confidence_threshold", 0.5, 0.9, step=0.05),
        }

    @classmethod
    def get_space(cls, strategy: str) -> Callable:
        """獲取策略對應的搜索空間"""
        spaces = {
            "momentum": cls.momentum_space,
            "mean_reversion": cls.mean_reversion_space,
            "trend_following": cls.trend_following_space,
            "ml_model": cls.ml_model_space,
        }
        return spaces.get(strategy, cls.momentum_space)


class LossFunctions:
    """損失函數實現"""

    @staticmethod
    def calculate_sharpe(
        results: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        計算夏普比率
        借鑒自 Freqtrade hyperopt_loss_sharpe.py

        Args:
            results: 交易結果 DataFrame（需包含 profit_ratio 列）
            risk_free_rate: 年化無風險利率
            periods_per_year: 每年交易期數

        Returns:
            夏普比率
        """
        if len(results) == 0 or 'profit_ratio' not in results.columns:
            return 0.0

        returns = results['profit_ratio']
        if len(returns) == 0:
            return 0.0

        # 日化無風險利率
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        return float(sharpe)

    @staticmethod
    def calculate_sortino(
        results: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        計算索提諾比率（只考慮下行風險）
        借鑒自 Freqtrade hyperopt_loss_sortino.py

        Args:
            results: 交易結果 DataFrame
            risk_free_rate: 年化無風險利率
            periods_per_year: 每年交易期數

        Returns:
            索提諾比率
        """
        if len(results) == 0 or 'profit_ratio' not in results.columns:
            return 0.0

        returns = results['profit_ratio']
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf

        # 只計算下行標準差
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            # 沒有虧損交易
            return excess_returns.mean() * np.sqrt(periods_per_year) if excess_returns.mean() > 0 else 0.0

        if downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        return float(sortino)

    @staticmethod
    def calculate_max_drawdown(
        results: pd.DataFrame,
        starting_balance: float
    ) -> float:
        """
        計算最大回撤

        Args:
            results: 交易結果 DataFrame
            starting_balance: 起始資金

        Returns:
            最大回撤（相對值）
        """
        if len(results) == 0 or 'profit_abs' not in results.columns:
            return 0.0

        # 計算累計收益
        cumulative = starting_balance + results['profit_abs'].cumsum()

        # 計算高點
        running_max = cumulative.cummax()

        # 計算回撤
        drawdown = (running_max - cumulative) / running_max

        return float(drawdown.max())

    @staticmethod
    def calculate_profit_drawdown_loss(
        results: pd.DataFrame,
        starting_balance: float,
        drawdown_mult: float = 0.075
    ) -> float:
        """
        計算利潤/回撤損失
        借鑒自 Freqtrade hyperopt_loss_profit_drawdown.py

        Args:
            results: 交易結果 DataFrame
            starting_balance: 起始資金
            drawdown_mult: 回撤懲罰係數

        Returns:
            損失值（越小越好）
        """
        if len(results) == 0 or 'profit_abs' not in results.columns:
            return 0.0

        total_profit = results['profit_abs'].sum()
        max_drawdown = LossFunctions.calculate_max_drawdown(results, starting_balance)

        # 損失 = -(利潤 - 回撤懲罰)
        # 回撤懲罰 = (回撤 * 利潤) * (1 - drawdown_mult)
        loss = -1 * (total_profit - (max_drawdown * total_profit) * (1 - drawdown_mult))

        return float(loss)

    @staticmethod
    def calculate_calmar(
        results: pd.DataFrame,
        starting_balance: float,
        periods_per_year: int = 252
    ) -> float:
        """
        計算 Calmar 比率（年化收益 / 最大回撤）

        Args:
            results: 交易結果 DataFrame
            starting_balance: 起始資金
            periods_per_year: 每年交易期數

        Returns:
            Calmar 比率
        """
        if len(results) == 0 or 'profit_abs' not in results.columns:
            return 0.0

        total_profit = results['profit_abs'].sum()
        max_drawdown = LossFunctions.calculate_max_drawdown(results, starting_balance)

        if max_drawdown == 0:
            return total_profit / starting_balance * periods_per_year if total_profit > 0 else 0.0

        # 計算年化收益率
        total_return = total_profit / starting_balance
        annualized_return = total_return * (periods_per_year / len(results))

        calmar = annualized_return / max_drawdown
        return float(calmar)


class HyperoptOptimizer:
    """Hyperopt 超參數優化器"""

    def __init__(self, config: Optional[HyperoptConfig] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperopt. Install with: pip install optuna")

        self.config = config or HyperoptConfig()
        self._study: Optional[optuna.Study] = None
        self._is_running = False
        self._current_trial = 0

        # 結果存儲
        self.results_dir = Path("models/hyperopt")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.optimization_history: List[OptimizationResult] = []

        logger.info("HyperoptOptimizer initialized")

    def update_config(self, **kwargs) -> HyperoptConfig:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                if key == "loss_function" and isinstance(value, str):
                    value = LossFunction(value)
                setattr(self.config, key, value)

        logger.info(f"HyperoptOptimizer config updated: {kwargs}")
        return self.config

    def get_config(self) -> Dict[str, Any]:
        """獲取配置"""
        return self.config.to_dict()

    def _calculate_loss(
        self,
        results: pd.DataFrame,
        loss_function: LossFunction
    ) -> float:
        """計算損失值"""
        if loss_function == LossFunction.SHARPE:
            return -LossFunctions.calculate_sharpe(
                results,
                risk_free_rate=self.config.risk_free_rate
            )

        elif loss_function == LossFunction.SORTINO:
            return -LossFunctions.calculate_sortino(
                results,
                risk_free_rate=self.config.risk_free_rate
            )

        elif loss_function == LossFunction.PROFIT_DRAWDOWN:
            return LossFunctions.calculate_profit_drawdown_loss(
                results,
                starting_balance=self.config.starting_balance,
                drawdown_mult=self.config.drawdown_mult
            )

        elif loss_function == LossFunction.CALMAR:
            return -LossFunctions.calculate_calmar(
                results,
                starting_balance=self.config.starting_balance
            )

        elif loss_function == LossFunction.TOTAL_PROFIT:
            if 'profit_abs' in results.columns:
                return -results['profit_abs'].sum()
            return 0.0

        return 0.0

    async def _run_backtest_with_params(
        self,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """使用給定參數運行回測"""
        # 這裡需要整合你現有的回測引擎
        # 示例實現：
        try:
            result = await backtest_engine.run_backtest(
                strategy_name=self.config.strategy,
                symbol=self.config.symbol,
                interval=self.config.interval,
                start_date=(datetime.now() - timedelta(days=self.config.data_days)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                initial_capital=self.config.starting_balance,
                **params
            )

            # 轉換為 DataFrame
            if isinstance(result, dict) and 'trades' in result:
                return pd.DataFrame(result['trades'])
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return pd.DataFrame()

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna 目標函數"""
        self._current_trial = trial.number + 1

        # 獲取搜索空間
        space_func = SearchSpace.get_space(self.config.strategy)
        params = space_func(trial)

        # 運行回測
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self._run_backtest_with_params(params))
        finally:
            loop.close()

        # 計算損失
        if len(results) == 0:
            return float('inf')  # 無效結果

        loss = self._calculate_loss(results, self.config.loss_function)

        # 記錄額外指標
        if len(results) > 0:
            trial.set_user_attr("trade_count", len(results))
            if 'profit_abs' in results.columns:
                trial.set_user_attr("total_profit", float(results['profit_abs'].sum()))
            if 'profit_ratio' in results.columns:
                trial.set_user_attr("avg_profit_ratio", float(results['profit_ratio'].mean()))

        return loss

    async def optimize(
        self,
        strategy: Optional[str] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> OptimizationResult:
        """
        執行超參數優化

        Args:
            strategy: 策略名稱
            n_trials: 試驗次數
            timeout: 超時時間（秒）

        Returns:
            優化結果
        """
        if self._is_running:
            raise RuntimeError("Optimization already running")

        strategy = strategy or self.config.strategy
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout

        self._is_running = True
        start_time = datetime.now()

        try:
            logger.info(f"Starting hyperopt for {strategy} with {n_trials} trials")

            # 創建 Study
            self._study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )

            # 運行優化
            self._study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.config.n_jobs,
                show_progress_bar=True,
                catch=(Exception,)
            )

            # 獲取結果
            best_params = self._study.best_params
            best_value = self._study.best_value
            total_time = (datetime.now() - start_time).total_seconds()

            # 構建試驗歷史
            all_trials = []
            for trial in self._study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    all_trials.append({
                        "number": trial.number,
                        "params": trial.params,
                        "value": trial.value,
                        "user_attrs": trial.user_attrs
                    })

            result = OptimizationResult(
                best_params=best_params,
                best_value=best_value,
                n_trials=len(self._study.trials),
                total_time=total_time,
                loss_function=self.config.loss_function.value,
                strategy=strategy,
                all_trials=all_trials
            )

            # 保存結果
            self._save_result(result)
            self.optimization_history.append(result)

            logger.info(f"Hyperopt completed. Best params: {best_params}, Best value: {best_value}")

            return result

        finally:
            self._is_running = False
            self._current_trial = 0

    def _save_result(self, result: OptimizationResult):
        """保存優化結果"""
        filename = f"hyperopt_{result.strategy}_{result.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Hyperopt result saved to {filepath}")

    def get_status(self) -> Dict[str, Any]:
        """獲取優化狀態"""
        return {
            "is_running": self._is_running,
            "current_trial": self._current_trial,
            "total_trials": self.config.n_trials,
            "config": self.config.to_dict(),
            "study_info": {
                "n_trials": len(self._study.trials) if self._study else 0,
                "best_value": self._study.best_value if self._study and self._study.best_trial else None,
                "best_params": self._study.best_params if self._study and self._study.best_trial else None
            } if self._study else None
        }

    def get_history(self, limit: int = 20) -> List[Dict]:
        """獲取優化歷史"""
        return [r.to_dict() for r in self.optimization_history[-limit:]]

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """獲取最佳參數"""
        if not self._study or not self._study.best_trial:
            return None
        return self._study.best_params

    def load_results(self) -> List[Dict]:
        """載入保存的優化結果"""
        results = []
        for filepath in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        return results

    def visualize_optimization(self) -> Dict[str, Any]:
        """生成優化可視化數據"""
        if not self._study:
            return {"error": "No study available"}

        try:
            # 參數重要性
            importance = optuna.importance.get_param_importances(self._study)

            # 優化歷史
            history = []
            for trial in self._study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    history.append({
                        "trial": trial.number,
                        "value": trial.value
                    })

            return {
                "param_importance": importance,
                "optimization_history": history,
                "best_trial": {
                    "number": self._study.best_trial.number,
                    "value": self._study.best_value,
                    "params": self._study.best_params
                }
            }

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return {"error": str(e)}


# 全局實例
hyperopt_optimizer = HyperoptOptimizer() if OPTUNA_AVAILABLE else None
