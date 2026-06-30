"""
影子交易執行器

定時檢測 SHADOW_RUNNING 狀態的模型，獲取市場數據，
讓模型產生預測信號，並記錄影子交易。

這個服務解決了模型註冊後但沒有實際交易記錄的問題。
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.logger import logger
from src.config import settings
from src.services.binance_client import binance_client
from src.services.arena_service import arena_service
from src.services.virtual_trading_service import virtual_trading_service
from src.engines.ml_trainer import ml_trainer, FeatureEngineer


class ShadowTradingExecutor:
    """影子交易執行器"""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_interval = 60  # 每 60 秒檢查一次
        self._models_cache: Dict[str, Any] = {}  # 緩存已載入的模型
        self._last_signals: Dict[str, Dict] = {}  # 記錄上次信號防止重複

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self):
        """啟動影子交易執行器"""
        if self._running:
            logger.warning("Shadow trading executor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Shadow trading executor started")

    async def stop(self):
        """停止影子交易執行器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Shadow trading executor stopped")

    async def _run_loop(self):
        """主循環"""
        while self._running:
            try:
                await self._execute_shadow_trades()
            except Exception as e:
                logger.error(f"Shadow trading loop error: {e}", exc_info=True)

            await asyncio.sleep(self._check_interval)

    async def _execute_shadow_trades(self):
        """執行影子交易檢測"""
        # 1. 獲取所有 SHADOW_RUNNING 狀態的模型
        shadow_models = await self._get_shadow_running_models()

        if not shadow_models:
            return

        logger.debug(f"Found {len(shadow_models)} shadow running models")

        # 2. 按交易對分組模型
        models_by_symbol = self._group_models_by_symbol(shadow_models)

        # 3. 對每個交易對執行預測
        for symbol, models in models_by_symbol.items():
            try:
                await self._process_symbol(symbol, models)
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")

    async def _get_shadow_running_models(self) -> List[Dict]:
        """獲取所有影子交易中的模型"""
        try:
            models = await arena_service.list_models(status='SHADOW_RUNNING')
            return [m.to_dict() if hasattr(m, 'to_dict') else m for m in models]
        except Exception as e:
            logger.error(f"Failed to get shadow models: {e}")
            return []

    def _group_models_by_symbol(self, models: List[Dict]) -> Dict[str, List[Dict]]:
        """按交易對分組模型"""
        groups = {}
        for model in models:
            # 從 version 中提取 symbol，例如 "ios-bnbusdt-random_forest-63" -> "BNBUSDT"
            version = model.get('version', '')
            parts = version.split('-')
            if len(parts) >= 2:
                symbol = parts[1].upper()  # bnbusdt -> BNBUSDT
            else:
                symbol = 'BTCUSDT'  # 預設

            if symbol not in groups:
                groups[symbol] = []
            groups[symbol].append(model)

        return groups

    async def _process_symbol(self, symbol: str, models: List[Dict]):
        """處理單個交易對的所有模型"""
        try:
            # 1. 獲取市場數據
            klines = await binance_client.get_klines(symbol, interval="15m", limit=100)

            if not klines:
                logger.warning(f"No klines data for {symbol}")
                return

            # 2. 轉換為 DataFrame
            df = self._klines_to_dataframe(klines)

            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                return

            # 3. 創建特徵
            features = FeatureEngineer.create_features(df)

            # 移除 NaN
            features = features.dropna()

            if features.empty:
                logger.warning(f"No valid features for {symbol}")
                return

            # 4. 獲取當前價格
            current_price = Decimal(str(df['close'].iloc[-1]))

            # 5. 更新所有模型的持倉價格（追蹤未實現盈虧）
            updated_count = await virtual_trading_service.update_position_prices(symbol, current_price)
            if updated_count > 0:
                logger.debug(f"Updated {updated_count} positions for {symbol} @ ${current_price}")

            # 6. 對每個模型生成信號
            for model_info in models:
                try:
                    await self._generate_signal_for_model(
                        model_info=model_info,
                        symbol=symbol,
                        features=features,
                        current_price=current_price
                    )
                except Exception as e:
                    logger.error(f"Error generating signal for model {model_info.get('version')}: {e}")

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")

    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """將 K 線數據轉換為 DataFrame"""
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 轉換數據類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    async def _generate_signal_for_model(
        self,
        model_info: Dict,
        symbol: str,
        features: pd.DataFrame,
        current_price: Decimal
    ):
        """為單個模型生成信號"""
        version = model_info.get('version', '')
        model_type_str = model_info.get('type', 'random_forest')

        # 從 type 中提取實際的模型類型
        # 例如 "iOS-random_forest" -> "random_forest"
        if '-' in model_type_str:
            model_type = model_type_str.split('-')[-1]
        else:
            model_type = model_type_str

        # 嘗試載入模型
        model_loaded = await self._ensure_model_loaded(version, model_type)

        if not model_loaded:
            # 如果沒有訓練好的模型，使用技術分析作為後備
            signal = await self._generate_technical_signal(features)
        else:
            # 使用 ML 模型預測
            signal = await self._generate_ml_signal(features, model_type)

        if signal is None:
            return

        # 檢查是否是新信號（防止重複記錄）
        signal_key = f"{version}:{symbol}"
        if self._is_duplicate_signal(signal_key, signal):
            return

        # 記錄影子交易
        await self._record_shadow_trade(
            model_version=version,
            symbol=symbol,
            signal=signal,
            current_price=current_price
        )

        # 更新模型統計
        await self._update_model_stats(version, signal)

    async def _ensure_model_loaded(self, version: str, model_type: str) -> bool:
        """確保模型已載入"""
        cache_key = f"{version}:{model_type}"

        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        # 嘗試載入模型
        try:
            # 構建模型路徑
            model_dir = Path(settings.MODEL_PATH if hasattr(settings, 'MODEL_PATH') else 'models')
            model_path = model_dir / f"{model_type}_model.pkl"

            if model_path.exists():
                success = ml_trainer.load_model(str(model_path), model_type)
                self._models_cache[cache_key] = success
                return success
            else:
                self._models_cache[cache_key] = False
                return False
        except Exception as e:
            logger.debug(f"Model not available for {version}: {e}")
            self._models_cache[cache_key] = False
            return False

    async def _generate_technical_signal(self, features: pd.DataFrame) -> Optional[Dict]:
        """使用技術分析生成信號（ML 模型不可用時的後備）"""
        try:
            if features.empty:
                return None

            latest = features.iloc[-1]

            # 簡單的技術分析信號
            signals = []

            # RSI 信號
            if 'rsi_14' in latest:
                rsi = latest['rsi_14']
                if rsi < 30:
                    signals.append(('BUY', 0.7))  # 超賣
                elif rsi > 70:
                    signals.append(('SELL', 0.7))  # 超買

            # MACD 信號
            if 'macd' in latest and 'macd_signal' in latest:
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                if macd > macd_signal:
                    signals.append(('BUY', 0.6))
                else:
                    signals.append(('SELL', 0.6))

            # 移動平均信號
            if 'sma_5' in latest and 'sma_20' in latest:
                if latest['sma_5'] > latest['sma_20']:
                    signals.append(('BUY', 0.5))
                else:
                    signals.append(('SELL', 0.5))

            if not signals:
                return None

            # 綜合信號
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']

            if len(buy_signals) > len(sell_signals):
                direction = 'BUY'
                confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            elif len(sell_signals) > len(buy_signals):
                direction = 'SELL'
                confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            else:
                return None  # 信號衝突，不交易

            # 計算信號強度
            strength = abs(len(buy_signals) - len(sell_signals)) / max(len(signals), 1)

            return {
                'direction': direction,
                'confidence': min(confidence, 0.95),
                'strength': min(strength, 1.0),
                'source': 'technical_analysis'
            }

        except Exception as e:
            logger.error(f"Technical signal generation error: {e}")
            return None

    async def _generate_ml_signal(
        self,
        features: pd.DataFrame,
        model_type: str
    ) -> Optional[Dict]:
        """使用 ML 模型生成信號"""
        try:
            # 取最後一行特徵進行預測
            latest_features = features.iloc[[-1]]

            predictions, probabilities = ml_trainer.predict(latest_features, model_type)

            if predictions is None or len(predictions) == 0:
                return None

            prediction = predictions[0]  # 0 = DOWN, 1 = UP

            # 獲取概率
            if probabilities is not None and len(probabilities) > 0:
                proba = probabilities[0]
                confidence = max(proba) if len(proba) > 1 else proba[0]
            else:
                confidence = 0.6  # 預設信心度

            direction = 'BUY' if prediction == 1 else 'SELL'

            return {
                'direction': direction,
                'confidence': float(confidence),
                'strength': float(confidence) * 0.9,  # 強度略低於信心度
                'source': f'ml_{model_type}'
            }

        except Exception as e:
            logger.error(f"ML signal generation error: {e}")
            return None

    def _is_duplicate_signal(self, signal_key: str, signal: Dict) -> bool:
        """檢查是否是重複信號"""
        if signal_key not in self._last_signals:
            self._last_signals[signal_key] = {
                'direction': signal['direction'],
                'timestamp': datetime.now()
            }
            return False

        last = self._last_signals[signal_key]

        # 如果方向相同且在 5 分鐘內，視為重複
        time_diff = (datetime.now() - last['timestamp']).total_seconds()
        if last['direction'] == signal['direction'] and time_diff < 300:
            return True

        # 更新記錄
        self._last_signals[signal_key] = {
            'direction': signal['direction'],
            'timestamp': datetime.now()
        }
        return False

    async def _record_shadow_trade(
        self,
        model_version: str,
        symbol: str,
        signal: Dict,
        current_price: Decimal
    ):
        """記錄影子交易並執行虛擬買賣"""
        try:
            # 獲取靈敏度配置
            sensitivity_config = await arena_service.get_sensitivity_config()

            confidence_threshold = sensitivity_config.confidence_threshold if hasattr(sensitivity_config, 'confidence_threshold') else 0.7
            strength_threshold = sensitivity_config.strength_threshold if hasattr(sensitivity_config, 'strength_threshold') else 0.6
            position_ratio = sensitivity_config.position_ratio if hasattr(sensitivity_config, 'position_ratio') else 0.6

            # 判斷是否會觸發交易
            would_trigger = (
                signal['confidence'] >= confidence_threshold and
                signal['strength'] >= strength_threshold
            )

            # ========== 執行虛擬交易 ==========
            if would_trigger:
                # 確保虛擬帳戶存在
                await virtual_trading_service.ensure_account(model_version)
                
                if signal['direction'] == 'BUY':
                    # 檢查是否已經有該 symbol 的持倉
                    existing_position = await virtual_trading_service.get_position(model_version, symbol)
                    
                    if not existing_position:
                        # 沒有持倉，執行買入
                        # 使用 10% 的餘額
                        account = await virtual_trading_service.get_account(model_version)
                        if account and account.balance > Decimal('10'):
                            quote_amount = account.balance * Decimal(str(position_ratio)) * Decimal('0.1')
                            quote_amount = min(quote_amount, Decimal('500'))  # 最多 500 USDT
                            
                            result = await virtual_trading_service.execute_virtual_buy(
                                model_version=model_version,
                                symbol=symbol,
                                price=current_price,
                                quote_amount=quote_amount,
                                signal_strength=signal['strength'],
                                signal_confidence=signal['confidence'],
                            )
                            
                            if result['success']:
                                logger.info(
                                    f"🟢 Virtual BUY executed: {model_version} | {symbol} | "
                                    f"qty={result['quantity']:.6f} @ ${current_price}"
                                )
                
                elif signal['direction'] == 'SELL':
                    # 檢查是否有持倉可以賣出
                    existing_position = await virtual_trading_service.get_position(model_version, symbol)
                    
                    if existing_position and existing_position.quantity > 0:
                        # 有持倉，執行賣出（全部賣出）
                        result = await virtual_trading_service.execute_virtual_sell(
                            model_version=model_version,
                            symbol=symbol,
                            price=current_price,
                            signal_strength=signal['strength'],
                            signal_confidence=signal['confidence'],
                        )
                        
                        if result['success']:
                            pnl_emoji = "🟢" if result.get('is_win') else "🔴"
                            logger.info(
                                f"{pnl_emoji} Virtual SELL executed: {model_version} | {symbol} | "
                                f"qty={result['quantity']:.6f} @ ${current_price} | "
                                f"PnL: ${result.get('realized_pnl', 0):.4f}"
                            )

            # 計算建議數量
            base_amount = Decimal('100')
            suggested_quantity = (base_amount * Decimal(str(position_ratio))) / current_price

            # 記錄影子交易到舊表（保持兼容性）
            success = await arena_service.record_shadow_trade(
                model_version=model_version,
                symbol=symbol,
                side=signal['direction'],
                signal_strength=signal['strength'],
                signal_confidence=signal['confidence'],
                suggested_quantity=suggested_quantity,
                suggested_price=current_price,
                actual_price=current_price,
                would_have_triggered=would_trigger,
                reason=f"Signal from {signal['source']}"
            )

            if success:
                logger.debug(
                    f"Shadow trade recorded: {model_version} | {symbol} | "
                    f"{signal['direction']} | conf={signal['confidence']:.2f} | "
                    f"would_trigger={would_trigger}"
                )

        except Exception as e:
            logger.error(f"Failed to record shadow trade: {e}", exc_info=True)

    async def _update_model_stats(self, version: str, signal: Dict):
        """更新模型統計數據"""
        try:
            # 獲取現有統計
            model = await arena_service.get_model(version)
            if not model:
                return

            # 更新交易計數 - 增加 trade_count
            current_count = model.trade_count if hasattr(model, 'trade_count') else 0

            # 更新模型統計
            await arena_service.update_model_stats(
                version=version,
                trade_count=current_count + 1
            )

        except Exception as e:
            logger.error(f"Failed to update model stats: {e}")

    def get_status(self) -> Dict[str, Any]:
        """獲取執行器狀態"""
        return {
            'running': self._running,
            'check_interval': self._check_interval,
            'cached_models': len(self._models_cache),
            'last_signals_tracked': len(self._last_signals)
        }


# 全局實例
shadow_trading_executor = ShadowTradingExecutor()
