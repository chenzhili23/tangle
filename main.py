# main.py
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Optional, Tuple, Set
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from sympy import symbols

# 假设 trendmaster.py 在同一目录下
from trendmaster import BinanceAPI


# 配置日志
def setup_logging():
    """配置交易日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trade.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('triangle_trader')


logger = setup_logging()


class PrecisionTriangleBreakoutTrader:
    """精准三角形突破交易策略 - 使用波段摆动点识别"""

    def __init__(self, api: BinanceAPI, symbols: List[str] = None, interval: str = "15m"):
        self.api = api
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        self.interval = interval

        # 策略参数
        self.leverage = 2
        self.stop_loss_pct = 0.005
        self.quick_profit_pct = 0.008
        self.min_volume_threshold = 5
        self.risk_per_trade = 0.95

        # 波段摆动参数
        self.swing_left_bars = 3  # 左边需要比较的K线数量
        self.swing_right_bars = 2  # 右边需要比较的K线数量
        self.min_swing_points = 3  # 最少需要3个波段点
        self.swing_confirmation_bars = 1  # 确认波段成立所需的后续K线数

        # 形态参数
        self.consolidation_ratio_threshold = 0.7

        # 精度信息缓存
        self.symbol_precision = {}

        # 状态变量
        self.active_position = None
        self.trading_enabled = True

    def get_symbol_precision(self, symbol: str) -> Dict:
        """获取交易对的精度信息"""
        if symbol in self.symbol_precision:
            return self.symbol_precision[symbol]

        try:
            # 获取交易对信息
            exchange_info = self.api.get_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    # 获取数量精度
                    lot_size_filter = next((f for f in s['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    if lot_size_filter:
                        step_size = float(lot_size_filter['stepSize'])
                        quantity_precision = self.get_precision_from_step(step_size)
                    else:
                        quantity_precision = 3  # 默认精度

                    # 获取价格精度
                    price_filter = next((f for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                    if price_filter:
                        tick_size = float(price_filter['tickSize'])
                        price_precision = self.get_precision_from_step(tick_size)
                    else:
                        price_precision = 2  # 默认精度

                    precision_info = {
                        'quantity': quantity_precision,
                        'price': price_precision
                    }

                    self.symbol_precision[symbol] = precision_info
                    logger.info(f"{symbol} 精度信息: 数量精度={quantity_precision}, 价格精度={price_precision}")
                    return precision_info

            # 如果没有找到，返回默认值
            default_precision = {'quantity': 3, 'price': 2}
            self.symbol_precision[symbol] = default_precision
            return default_precision

        except Exception as e:
            logger.error(f"获取{symbol}精度信息失败: {e}")
            default_precision = {'quantity': 3, 'price': 2}
            self.symbol_precision[symbol] = default_precision
            return default_precision

    def get_precision_from_step(self, step_size: float) -> int:
        """从步长计算精度"""
        if step_size >= 1:
            return 0
        step_str = str(step_size)
        if 'e-' in step_str:
            return int(step_str.split('e-')[1])
        if '.' in step_str:
            return len(step_str.split('.')[1].rstrip('0'))
        return 0

    def get_account_balance(self) -> float:
        """获取账户可用余额"""
        try:
            balance_info = self.api.get_balance()
            for asset in balance_info:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            return 1000.0  # 默认值

    def find_swing_points(self, highs: List[float], lows: List[float],
                          close: List[float], volume: List[float]) -> Tuple[List[int], List[int]]:
        """
        识别波段高点(Swing High)和波段低点(Swing Low) - 优化版本
        """
        swing_highs = []
        swing_lows = []

        length = len(highs)

        # 使用更宽松的参数
        left_bars = 2  # 从3降到2
        right_bars = 1  # 从2降到1

        # 增加确认K线数，提高准确性
        confirmation_bars = 2  # 从1增加到2

        # 从足够早的位置开始识别，确保有左右对比的空间
        start_idx = left_bars
        end_idx = length - right_bars - confirmation_bars

        for i in range(start_idx, end_idx):
            # 检查是否为波段高点 (Swing High)
            is_swing_high = True

            # 检查左边是否有更高的高点
            for j in range(1, left_bars + 1):
                if highs[i] < highs[i - j]:  # 改为严格小于
                    is_swing_high = False
                    break

            # 检查右边是否有更高的高点
            if is_swing_high:
                for j in range(1, right_bars + 1):
                    if highs[i] < highs[i + j]:  # 改为严格小于
                        is_swing_high = False
                        break

            # 检查是否为波段低点 (Swing Low)
            is_swing_low = True

            # 检查左边是否有更低的低点
            for j in range(1, left_bars + 1):
                if lows[i] > lows[i - j]:  # 改为严格大于
                    is_swing_low = False
                    break

            # 检查右边是否有更低的低点
            if is_swing_low:
                for j in range(1, right_bars + 1):
                    if lows[i] > lows[i + j]:  # 改为严格大于
                        is_swing_low = False
                        break

            # 确认波段点的有效性（等待确认K线）
            if is_swing_high and self.confirm_swing_point(i, highs, lows, 'high', confirmation_bars):
                swing_highs.append(i)

            if is_swing_low and self.confirm_swing_point(i, highs, lows, 'low', confirmation_bars):
                swing_lows.append(i)

        return swing_highs, swing_lows

    def confirm_swing_point(self, idx: int, highs: List[float], lows: List[float],
                            point_type: str, confirmation_bars: int = 2) -> bool:
        """
        确认波段点是否有效 - 增加确认K线数
        """
        confirmation_start = idx + 1
        confirmation_end = idx + confirmation_bars

        if confirmation_end >= len(highs):
            return False

        if point_type == 'high':
            # 对于波段高点，确认后续K线没有突破该高点
            for i in range(confirmation_start, confirmation_end + 1):
                if highs[i] >= highs[idx]:  # 改为大于等于
                    return False
        else:  # 'low'
            # 对于波段低点，确认后续K线没有跌破该低点
            for i in range(confirmation_start, confirmation_end + 1):
                if lows[i] <= lows[idx]:  # 改为小于等于
                    return False

        return True

    def is_triangle_converging(self, upper_line: List[float], lower_line: List[float],
                               ratio_threshold: float = 0.8) -> bool:
        """检查是否为收敛三角形 - 使用可配置的阈值"""
        if len(upper_line) < 10 or len(lower_line) < 10:
            return False

        # 检查趋势线是否收敛
        early_diff = upper_line[10] - lower_line[10]
        late_diff = upper_line[-1] - lower_line[-1]

        # 避免除以零
        if abs(early_diff) < 1e-10:
            return False

        convergence_ratio = late_diff / early_diff

        return convergence_ratio < ratio_threshold

    def filter_significant_swings(self, swing_points: List[int], prices: List[float],
                                  min_price_change: float = 0.005) -> List[int]:
        """
        过滤掉幅度太小的波段点
        min_price_change: 最小价格变化百分比（0.5%）
        """
        if len(swing_points) < 2:
            return swing_points

        filtered_points = [swing_points[0]]

        for i in range(1, len(swing_points)):
            current_idx = swing_points[i]
            prev_idx = filtered_points[-1]

            price_change = abs(prices[current_idx] - prices[prev_idx]) / prices[prev_idx]

            if price_change >= min_price_change:
                filtered_points.append(current_idx)
                logger.debug(f"有效波段点: 位置 {current_idx}, 价格 {prices[current_idx]}, 变化 {price_change:.3%}")

        return filtered_points

    def calculate_precision_trendlines(self, symbol: str, klines: List) -> Optional[Dict]:
        """使用波段摆动点计算三角形形态和突破点 - 优化版本"""
        if len(klines) < 30:  # 减少所需的最小K线数量，从50降到30
            logger.debug(f"{symbol} 数据不足，需要至少30根K线，当前只有{len(klines)}根")
            return None

        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        # 增加调试信息
        logger.debug(f"{symbol} 数据分析: 最高价范围 [{min(highs):.4f}, {max(highs):.4f}], "
                     f"最低价范围 [{min(lows):.4f}, {max(lows):.4f}]")

        # 寻找波段高点和低点 - 使用更宽松的参数
        swing_highs, swing_lows = self.find_swing_points(highs, lows, closes, volumes)

        # 增加调试日志
        print(f"{symbol} 初步识别到 {len(swing_highs)} 个波段高点和 {len(swing_lows)} 个波段低点")

        # 过滤掉幅度太小的波段点 - 使用更宽松的阈值
        significant_highs = self.filter_significant_swings(swing_highs, highs, min_price_change=0.003)  # 从0.005降到0.003
        significant_lows = self.filter_significant_swings(swing_lows, lows, min_price_change=0.003)

        print(f"{symbol} 识别到 {len(significant_highs)} 个显著波段高点和 {len(significant_lows)} 个显著波段低点")

        # 如果有波段点，打印它们的价格和位置
        if significant_highs:
            logger.debug(f"显著高点位置和价格: {[(idx, highs[idx]) for idx in significant_highs]}")
        if significant_lows:
            logger.debug(f"显著低点位置和价格: {[(idx, lows[idx]) for idx in significant_lows]}")

        # 降低最小波段点要求，从3降到2
        min_points_required = 2
        if len(significant_highs) < min_points_required or len(significant_lows) < min_points_required:
            logger.debug(f"{symbol} 波段点不足: 需要至少{min_points_required}个高点和低点, "
                         f"实际有{len(significant_highs)}个高点和{len(significant_lows)}个低点")
            return None

        # 计算上下趋势线 - 使用所有有效的波段点
        upper_line = self.calculate_swing_based_trendline(significant_highs, highs, len(highs), 'upper')
        lower_line = self.calculate_swing_based_trendline(significant_lows, lows, len(lows), 'lower')

        if not upper_line or not lower_line:
            logger.debug(f"{symbol} 无法计算有效的趋势线")
            return None

        # 检查三角形收敛形态 - 使用更宽松的条件
        if not self.is_triangle_converging(upper_line, lower_line, ratio_threshold=0.8):  # 从0.7增加到0.8
            logger.debug(f"{symbol} 形态未收敛，不构成三角形")
            return None

        current_price = closes[-1]
        current_upper = upper_line[-1] if upper_line else 0
        current_lower = lower_line[-1] if lower_line else 0

        logger.debug(f"{symbol} 当前价格: {current_price:.4f}, 上轨: {current_upper:.4f}, 下轨: {current_lower:.4f}")

        # 确定突破方向和突破点
        breakout_info = self.detect_breakout_point(
            current_price, current_upper, current_lower,
            highs, lows, volumes
        )

        if breakout_info:
            logger.info(f"✅ {symbol} 发现突破信号! 方向: {breakout_info['direction']}, "
                        f"价格: {breakout_info['breakout_price']:.4f}")
            breakout_info.update({
                'symbol': symbol,
                'current_price': current_price,
                'upper_trendline': current_upper,
                'lower_trendline': current_lower,
                'swing_highs': significant_highs[-4:],  # 最近4个波段高点
                'swing_lows': significant_lows[-4:],  # 最近4个波段低点
                'breakout_time': datetime.now()
            })
            return breakout_info
        else:
            logger.debug(f"{symbol} 未检测到突破点")

        return None

    def calculate_swing_based_trendline(self, swing_points: List[int], prices: List[float],
                                        data_length: int, trend_type: str) -> Optional[List[float]]:
        """
        基于波段点计算趋势线
        trend_type: 'upper' 或 'lower'
        """
        if len(swing_points) < 2:
            return None

        # 使用最近3-4个波段点进行拟合，以获得更稳定的趋势线
        recent_points = swing_points[-min(4, len(swing_points)):]

        x = np.array(recent_points)
        y = np.array([prices[i] for i in recent_points])

        try:
            # 使用线性回归拟合趋势线
            coeffs = np.polyfit(x, y, 1)
            trendline = [np.polyval(coeffs, i) for i in range(data_length)]

            # 验证趋势线的合理性
            if self.validate_trendline(coeffs, trend_type):
                return trendline
            else:
                return None

        except Exception as e:
            logger.error(f"{trend_type}趋势线计算错误: {e}")
            return None

    def validate_trendline(self, coeffs: np.ndarray, trend_type: str) -> bool:
        """
        验证趋势线的合理性
        coeffs: [斜率, 截距]
        """
        slope = coeffs[0]

        if trend_type == 'upper':
            # 上轨通常应该有负斜率（下降）或轻微正斜率
            return slope < 0.0005  # 允许轻微正斜率但限制过大
        else:  # 'lower'
            # 下轨通常应该有正斜率（上升）或轻微负斜率
            return slope > -0.0005  # 允许轻微负斜率但限制过大



    def detect_breakout_point(self, current_price: float, upper_line: float,
                              lower_line: float, highs: List[float],
                              lows: List[float], volumes: List[float]) -> Optional[Dict]:
        """检测突破点"""
        # 突破上轨
        if current_price > upper_line and current_price > max(highs[-5:-1]):
            return {
                'direction': 'up',
                'breakout_price': current_price,
                'stop_loss': lower_line,  # b点下方作为止损
                'breakout_type': 'upper'
            }

        # 突破下轨
        elif current_price < lower_line and current_price < min(lows[-5:-1]):
            return {
                'direction': 'down',
                'breakout_price': current_price,
                'stop_loss': upper_line,  # b点上方作为止损
                'breakout_type': 'lower'
            }

        return None

    def analyze_order_book_strength(self, symbol: str, order_book: Dict, current_price: float) -> Dict:
        """分析盘口强度 - 基于相对挂单量比较，并打印详细挂单情况"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                print(f"{symbol} 盘口数据为空")
                return {'defense_weak': False, 'pressure_ratio': 0, 'relative_pressure': 0}

            # 打印完整的盘口情况
            self.print_order_book_details(symbol, asks, bids, current_price)

            # 获取精度信息，确定最小价格变动单位
            precision_info = self.get_symbol_precision(symbol)
            price_precision = precision_info['price']

            # 计算最小价格变动单位
            min_price_step = 10 ** (-price_precision)

            # 确定价格间隔 - 使用更大的间隔（例如10个最小变动单位）
            price_interval = min_price_step * 1000  # 10个最小价格变动单位

            # 分析间隔更大的档位
            key_ask_levels = []  # 当前价格上方的卖盘
            key_bid_levels = []  # 当前价格下方的买盘

            # 收集当前价格上方的卖盘，使用更大的价格间隔
            last_selected_price = None
            for price, qty in asks:
                price_float = float(price)

                # 如果是第一个档位或者价格间隔足够大
                if last_selected_price is None or (price_float - last_selected_price) >= price_interval:
                    if price_float > current_price and len(key_ask_levels) < 100:
                        key_ask_levels.append((price_float, float(qty)))
                        last_selected_price = price_float

                        # 如果已经收集到足够的档位，停止
                        if len(key_ask_levels) >= 100:
                            break

            # 收集当前价格下方的买盘，使用更大的价格间隔
            last_selected_price = None
            for price, qty in bids:
                price_float = float(price)

                # 如果是第一个档位或者价格间隔足够大
                if last_selected_price is None or (last_selected_price - price_float) >= price_interval:
                    if price_float < current_price and len(key_bid_levels) < 100:
                        key_bid_levels.append((price_float, float(qty)))
                        last_selected_price = price_float

                        # 如果已经收集到足够的档位，停止
                        if len(key_bid_levels) >= 100:
                            break

            # 计算相对挂单量
            ask_pressure = sum(qty for _, qty in key_ask_levels)
            bid_support = sum(qty for _, qty in key_bid_levels)

            # 动态阈值：基于盘口平均挂单量和价格调整
            dynamic_threshold = self.calculate_dynamic_threshold(symbol, asks, bids, current_price)

            # 相对压力比率
            pressure_ratio = ask_pressure / bid_support if bid_support > 0 else float('inf')
            relative_pressure = ask_pressure / dynamic_threshold if dynamic_threshold > 0 else float('inf')

            # 防守薄弱判断：卖压相对较小或压力比率较低
            defense_weak = (relative_pressure < 1.0 or pressure_ratio < 0.8)

            # 打印分析结果
            print(f"📊 {symbol} 盘口分析结果 (间隔 {price_interval}):")
            print(f"   当前价格: {current_price:.4f}")
            print(f"   最小价格变动: {min_price_step}")
            print(f"   分析价格间隔: {price_interval}")
            print(f"   上方卖压: {ask_pressure:.4f}")
            print(f"   下方买撑: {bid_support:.4f}")
            print(f"   动态阈值: {dynamic_threshold:.4f}")
            print(f"   压力比率(卖/买): {pressure_ratio:.3f}")
            print(f"   相对压力(卖/阈值): {relative_pressure:.3f}")
            print(f"   防守薄弱: {defense_weak}")

            # 打印具体的分析档位
            print("🔍 分析使用的档位:")
            for i, (price, qty) in enumerate(key_ask_levels):
                diff_percent = ((price - current_price) / current_price) * 100
                print(f"   卖档{i + 1}: {price:.4f} (+{diff_percent:.3f}%) - 数量: {qty:.4f}")
            for i, (price, qty) in enumerate(key_bid_levels):
                diff_percent = ((current_price - price) / current_price) * 100
                print(f"   买档{i + 1}: {price:.4f} (-{diff_percent:.3f}%) - 数量: {qty:.4f}")

            return {
                'defense_weak': defense_weak,
                'pressure_ratio': pressure_ratio,
                'relative_pressure': relative_pressure,
                'ask_pressure': ask_pressure,
                'bid_support': bid_support,
                'dynamic_threshold': dynamic_threshold,
                'key_ask_levels': key_ask_levels,
                'key_bid_levels': key_bid_levels
            }

        except Exception as e:
            logger.error(f"{symbol} 盘口分析错误: {e}")
            return {'defense_weak': False, 'pressure_ratio': 0, 'relative_pressure': 0}

    def print_order_book_details(self, symbol: str, asks: List, bids: List, current_price: float):
        """打印详细的盘口情况 - 显示更多档位"""
        print(f"🎯 {symbol} 当前价格: {current_price:.4f}")
        print("🔴 卖盘(Ask) - 上方压力:")
        for i, (price, qty) in enumerate(asks[:10]):  # 增加到前10档卖盘
            price_float = float(price)
            qty_float = float(qty)
            diff_percent = ((price_float - current_price) / current_price) * 100
            print(f"   卖{i + 1}: {price_float:.4f} (+{diff_percent:.2f}%) - 数量: {qty_float:.4f}")

        print("🟢 买盘(Bid) - 下方支撑:")
        for i, (price, qty) in enumerate(bids[:10]):  # 增加到前10档买盘
            price_float = float(price)
            qty_float = float(qty)
            diff_percent = ((current_price - price_float) / current_price) * 100
            print(f"   买{i + 1}: {price_float:.4f} (-{diff_percent:.2f}%) - 数量: {qty_float:.4f}")

        # 计算总买卖压力 - 增加到前200档
        total_ask_pressure = sum(float(qty) for _, qty in asks[:200])
        total_bid_support = sum(float(qty) for _, qty in bids[:200])
        print(f"📈 前200档总卖压: {total_ask_pressure:.4f}")
        print(f"📉 前200档总买撑: {total_bid_support:.4f}")
        print(
            f"⚖️  买卖比例: {total_ask_pressure / total_bid_support:.3f}" if total_bid_support > 0 else "⚖️  买卖比例: 无限大")

    def calculate_dynamic_threshold(self, symbol: str, asks: List, bids: List, current_price: float) -> float:
        """计算动态阈值：基于盘口平均挂单量和价格"""
        try:
            # 计算整个盘口的平均挂单量 - 增加到前300档
            all_asks_volume = sum(float(qty) for _, qty in asks[:300])
            all_bids_volume = sum(float(qty) for _, qty in bids[:300])

            avg_volume = (all_asks_volume + all_bids_volume) / 60 if (all_asks_volume + all_bids_volume) > 0 else 1.0

            # 基于价格调整阈值
            price_factor = max(0.1, min(1.0, current_price / 100))

            # 动态阈值 = 平均挂单量 × 价格因子 × 10（10个档位）
            dynamic_threshold = avg_volume * price_factor * 100

            return max(dynamic_threshold, 0.1)
        except:
            return 5.0

    def execute_breakout_trade(self, breakout_signal: Dict, order_book_analysis: Dict):
        """执行突破点交易 - 全仓模式"""
        symbol = breakout_signal['symbol']
        direction = breakout_signal['direction']
        entry_price = breakout_signal['breakout_price']
        stop_loss = breakout_signal['stop_loss']

        try:
            # 计算全仓仓位大小
            position_size = self.calculate_full_position_size(symbol, entry_price)

            if position_size <= 0:
                logger.warning(f"仓位大小计算为0或负数，跳过交易")
                return

            # 设置杠杆
            self.api.change_leverage(symbol, self.leverage)

            # 执行突破交易
            if direction == 'up':
                order = self.api.market_order(symbol=symbol, side="BUY", quantity=position_size)
                action = "突破做多"
            else:
                order = self.api.market_order(symbol=symbol, side="SELL", quantity=position_size)
                action = "突破做空"

            # 记录交易
            self.active_position = {
                'symbol': symbol,
                'size': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'direction': direction,
                'entry_time': datetime.now(),
                'order_id': order.get('orderId'),
                'breakout_info': breakout_signal
            }

            logger.info(f"🎯 {action} {symbol} @ {entry_price}")
            logger.info(f"止损: {stop_loss}, 仓位: {position_size}")
            # 发送通知
            self.api._send_feishu_notification_new(
                "全局交易入场",
                f"{action} {symbol}\n数量: {position_size}\n价格: {entry_price}\n杠杆: {self.leverage}x"
            )

            # 立即设置止盈止损订单
            self.set_quick_profit_target(symbol, direction, entry_price, position_size)

        except Exception as e:
            logger.error(f"突破交易执行失败: {e}")

    def calculate_full_position_size(self, symbol: str, entry_price: float) -> float:
        """计算全仓模式下的仓位大小，并确保精度正确"""
        try:
            # 获取账户余额
            account_balance = self.get_account_balance()

            # 计算可用资金（考虑风险比例）
            available_balance = account_balance * self.risk_per_trade

            # 计算基于杠杆的仓位价值
            position_value = available_balance * self.leverage

            # 计算仓位数量
            position_size = position_value / entry_price

            # 获取精度信息并调整仓位大小
            precision_info = self.get_symbol_precision(symbol)
            quantity_precision = precision_info['quantity']

            # 根据精度调整仓位大小
            adjusted_size = round(position_size, quantity_precision)

            logger.info(
                f"{symbol} 仓位计算: 余额={account_balance}, 计算数量={position_size}, 调整后数量={adjusted_size}")

            return adjusted_size

        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            return 0.0

    def set_quick_profit_target(self, symbol: str, direction: str, entry_price: float, size: float):
        """设置快速止盈目标，确保价格精度正确"""
        try:
            # 获取精度信息
            precision_info = self.get_symbol_precision(symbol)
            price_precision = precision_info['price']

            if direction == 'up':
                target_price = entry_price * (1 + self.quick_profit_pct)
                # 根据精度调整价格
                target_price = round(target_price, price_precision)
                # 设置限价止盈单
                self.api.limit_order(symbol=symbol, side="SELL", quantity=size, price=target_price)
            else:
                target_price = entry_price * (1 - self.quick_profit_pct)
                # 根据精度调整价格
                target_price = round(target_price, price_precision)
                # 设置限价止盈单
                self.api.limit_order(symbol=symbol, side="BUY", quantity=size, price=target_price)

            logger.info(f"设置止盈目标: {target_price} (精度: {price_precision})")
        except Exception as e:
            logger.error(f"设置止盈目标失败: {e}")

    def monitor_active_position(self):
        """监控当前持仓"""
        if not self.active_position:
            return

        symbol = self.active_position['symbol']

        try:
            # 获取当前价格
            klines = self.api.get_klines(symbol=symbol, interval=self.interval, limit=5)
            current_price = float(klines[-1][4])

            # 检查是否达到止损
            if self.active_position['direction'] == 'up':
                if current_price <= self.active_position['stop_loss']:
                    logger.info(f"触发止损，平仓 {symbol}")
                    self.close_position()
            else:
                if current_price >= self.active_position['stop_loss']:
                    logger.info(f"触发止损，平仓 {symbol}")
                    self.close_position()

        except Exception as e:
            logger.error(f"监控持仓失败: {e}")

    def close_position(self):
        """平仓当前持仓"""
        if not self.active_position:
            return

        symbol = self.active_position['symbol']
        position_size = self.active_position['size']
        direction = self.active_position['direction']

        try:
            if direction == 'up':
                # 平多仓
                order = self.api.market_order(symbol=symbol, side="SELL", quantity=position_size)
            else:
                # 平空仓
                order = self.api.market_order(symbol=symbol, side="BUY", quantity=position_size)

            logger.info(f"已平仓 {symbol}, 数量: {position_size}")
            self.active_position = None

        except Exception as e:
            logger.error(f"平仓失败: {e}")

    def run_precision_strategy(self):
        """运行精准突破策略"""
        logger.info("🚀 启动精准三角形突破策略")
        balance = self.get_account_balance()
        logger.info(f'当前余额: {balance} USDT')
        logger.info(f'全仓模式: 每笔交易使用 {self.risk_per_trade * 100}% 的资金')

        # 预先加载所有交易对的精度信息
        for symbol in self.symbols:
            self.get_symbol_precision(symbol)

        while True:
            try:
                if self.active_position:
                    self.monitor_active_position()
                    time.sleep(10)
                    continue

                # 检查所有币种的突破机会
                for symbol in self.symbols:
                    if not self.trading_enabled:
                        break

                    # 获取K线数据和盘口
                    klines = self.api.get_klines(symbol=symbol, interval=self.interval, limit=40)
                    order_book = self.api.get_order_book(symbol=symbol, limit=20)
                    current_price = float(klines[-1][4])

                    # 分析突破信号和盘口强度
                    breakout_signal = self.calculate_precision_trendlines(symbol, klines)
                    ob_analysis = self.analyze_order_book_strength(symbol, order_book, current_price)

                    if breakout_signal and ob_analysis['defense_weak']:
                        logger.info(f"✅ {symbol} 发现突破机会!")
                        self.execute_breakout_trade(breakout_signal, ob_analysis)
                        break

                time.sleep(10)

            except Exception as e:
                logger.error(f"策略执行错误: {e}")
                time.sleep(2)


def main():
    # 从配置文件或环境变量读取API密钥
    API_KEY = "KJmN1lS3U6KCIU1r2qh4af7IXYWv2GdryrHMAWa5PQEeQPYb2ZwR6l5yB7ZB9UAQ"
    API_SECRET = "iLpBAQm0z0sCV7YET0FI1dJqbDga8NrIAm7snloUQRVlkocSKnPa98lEXXRwBFNF"

    # 初始化API
    api = BinanceAPI(API_KEY, API_SECRET)

    # 支持的币种列表
    target_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
                      "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]

    # 初始化交易策略
    trader = PrecisionTriangleBreakoutTrader(api)

    trader.run_precision_strategy()


if __name__ == "__main__":
    main()