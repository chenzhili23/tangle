from datetime import datetime, timedelta

import requests
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import joblib
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
import json



class BinanceAPI:
    """专门处理币安合约交易的API类"""

    def __init__(self, key, secret):
        self.BASE_URL = "https://fapi.binance.com"
        self.key = key
        self.secret = secret
        self.feishu_webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/d4ade6d6-f3e0-451f-82e7-581911517ba4"#个人

    def _sign(self, params={}):
        """生成签名"""
        data = params.copy()
        ts = int(1000 * time.time())
        data.update({"timestamp": ts})
        h = urlencode(data)
        b = bytearray()
        b.extend(self.secret.encode())
        signature = hmac.new(b, msg=h.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
        data.update({"signature": signature})
        return data

    def _request(self, method, path, params={}):
        """发送请求的底层方法"""
        params.update({"recvWindow": 6000})
        query = self._sign(params)
        headers = {"X-MBX-APIKEY": self.key}

        if method == "GET":
            url = f"{self.BASE_URL}{path}?{urlencode(query)}"
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            url = f"{self.BASE_URL}{path}"
            response = requests.post(url, headers=headers, data=query, timeout=30)
        elif method == "DELETE":
            url = f"{self.BASE_URL}{path}?{urlencode(query)}"
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError("Unsupported HTTP method")

        # Print response text if status code is 400
        if response.status_code == 400:
            print(f"Error 400: {response.text}")
            self._send_feishu_notification("请求失败", response.text)

        response.raise_for_status()
        return response.json()

    def get_position_risk(self):
        """获取用户持仓风险信息 (GET /fapi/v2/positionRisk)"""
        endpoint = "/fapi/v2/positionRisk"
        return self._request("GET", endpoint)

    def get_open_interest(self, symbol):
        """获取未平仓量 (GET /fapi/v1/openInterest)"""
        endpoint = "/fapi/v1/openInterest"
        params = {'symbol': symbol}
        return self._request("GET", endpoint, params)

    def _send_feishu_notification_new(self, title, content):
        """发送飞书通知"""
        beijing_time = datetime.now() + timedelta(hours=8)
        formatted_time = beijing_time.strftime('%Y-%m-%d %H:%M:%S')
        headers = {'Content-Type': 'application/json'}
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "content": title,
                        "tag": "plain_text"
                    }
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "content": content,
                            "tag": "plain_text"
                        }
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"北京时间: {formatted_time}"
                            }
                        ]
                    }
                ]
            }
        }

        try:
            response = requests.post(self.feishu_webhook, headers=headers, json=payload)
            response.raise_for_status()
        except Exception as e:
            print(f"发送飞书通知失败: {str(e)}")

    # ------------ 账户相关 ------------
    def get_account_info(self):
        """获取合约账户信息"""
        return self._request("GET", "/fapi/v3/account")

    def get_balance(self):
        """获取合约账户余额"""
        return self._request("GET", "/fapi/v3/balance")

    # ------------ 交易相关 ------------
    def create_order(self, symbol, side, order_type, quantity,
                     price=None, time_in_force=None, reduce_only=False,
                     stop_price=None, close_position=False):
        """
        创建合约订单
        :param symbol: 交易对，如BTCUSDT
        :param side: 买卖方向 BUY/SELL
        :param position_side: 持仓方向 LONG/SHORT
        :param order_type: 订单类型 LIMIT/MARKET/STOP/TAKE_PROFIT等
        :param quantity: 数量
        :param price: 价格(限价单需要)
        :param time_in_force: 有效时间(GTC,IOC,FOK)
        :param reduce_only: 是否只减仓
        :param stop_price: 触发价(条件订单需要)
        :param close_position: 是否平仓
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "closePosition": str(close_position).lower()
        }

        if price is not None:
            params["price"] = price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if stop_price is not None:
            params["stopPrice"] = stop_price

        return self._request("POST", "/fapi/v1/order", params)

    def market_order(self, symbol, side, quantity):
        """市价单"""
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity
        )

    def limit_order(self, symbol, side, quantity, price, time_in_force="GTC"):
        """限价单"""
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=price,
            time_in_force=time_in_force
        )

    def stop_market_order(self, symbol, side, quantity, stop_price):
        """止损市价单"""
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type="STOP_MARKET",
            quantity=quantity,
            stop_price=stop_price
        )

    def cancel_order(self, symbol, order_id=None, orig_client_order_id=None):
        """取消订单"""
        params = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        return self._request("DELETE", "/fapi/v1/order", params)

    def cancel_all_orders(self, symbol):
        """取消所有订单"""
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})

    # ------------ 持仓相关 ------------
    def get_position_info(self, symbol=None):
        """获取持仓信息"""
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v3/positionRisk", params)

    def change_leverage(self, symbol, leverage):
        """调整杠杆"""
        return self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage
        })

    def change_margin_type(self, symbol, margin_type):
        """调整保证金模式"""
        return self._request("POST", "/fapi/v1/marginType", {
            "symbol": symbol,
            "marginType": margin_type
        })

    # ------------ 行情相关 ------------
    def get_price(self, symbol=None):
        """获取最新价格"""
        path = "/fapi/v1/ticker/price"
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        return self._request("GET", path, params)

    def get_order_book(self, symbol, limit=100):
        """获取深度数据"""
        return self._request("GET", "/fapi/v1/depth", {
            "symbol": symbol,
            "limit": limit
        })

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """获取K线数据"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return self._request("GET", "/fapi/v1/klines", params)

    # ------------ 高级功能 ------------
    def batch_orders(self, orders):
        """批量下单"""
        return self._request("POST", "/fapi/v1/batchOrders", {
            "batchOrders": json.dumps(orders)
        })

    def get_open_orders(self, symbol=None):
        """获取当前挂单"""
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openOrders", params)

    def get_order_history(self, symbol, start_time=None, end_time=None, limit=500):
        """获取历史订单"""
        params = {"symbol": symbol}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", "/fapi/v1/allOrders", params)

    def get_trades(self, symbol, start_time=None, end_time=None, limit=500):
        """获取成交历史"""
        params = {"symbol": symbol}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", "/fapi/v1/userTrades", params)

    def get_funding_rate_history(self, symbol=None, start_time=None, end_time=None, limit=100):
        """获取资金费率历史"""
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", "/fapi/v1/fundingRate", params)

    def get_exchange_info(self, symbol=None):
        """获取交易对的详细信息（包括交易规则、精度等）

        :param symbol: 交易对符号，如"BTCUSDT"。如果为None则返回所有交易对信息
        :return: 包含交易对信息的字典
        """
        params = {}
        if symbol is not None:
            params['symbol'] = symbol

        return self._request("GET", "/fapi/v1/exchangeInfo", params)





# ---------------------- DataLoader ----------------------

class DataLoader:
    """DataLoader class for fetching and preprocessing stock/crypto data."""

    def __init__(self, data_dir="data"):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        # 为每个特征创建单独的scaler
    def _get_data_filename(self, symbol, interval):
        """Generate standardized filename for storing data."""
        return os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{interval}_data.csv")

    def get_historical_data(self, symbol, interval, start_date=None, end_date=None, limit=1000, force_download=False):
        """Fetch historical data with enhanced storage (batched to handle API limits)."""
        filename = self._get_data_filename(symbol, interval)
        print(f"获取历史数据{filename}")

        # Try to load from cache if not forcing download
        if not force_download and os.path.exists(filename):
            try:
                data = pd.read_csv(filename, index_col='Date', parse_dates=True)
                print(f"Loaded cached data for {symbol} {interval} from {filename}")
                return data
            except Exception as e:
                print(f"Error loading cached data: {e}, re-downloading...")

        # Convert string dates to datetime if needed
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Initialize empty DataFrame to store all data
        all_data = pd.DataFrame()

        # Binance API limit (1000 candles per request)
        batch_size = min(limit, 1000) if limit else 1000
        current_start = start_date

        base_url = "https://api.binance.com/api/v3/klines"
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            while current_start < end_date:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': batch_size,
                    'startTime': int(current_start.timestamp() * 1000),
                    'endTime': int(end_date.timestamp() * 1000)
                }

                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                batch_data = response.json()

                if not batch_data:
                    print(f"No more data available after {current_start}")
                    break

                # Convert batch to DataFrame
                columns = [
                    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                ]

                df_batch = pd.DataFrame(batch_data, columns=columns, dtype=float)
                df_batch['Date'] = pd.to_datetime(df_batch['Open time'], unit='ms')
                df_batch.set_index('Date', inplace=True)
                df_batch = df_batch[['Open', 'High', 'Low', 'Close', 'Volume']]

                # Concatenate with existing data
                all_data = pd.concat([all_data, df_batch])

                # Update current_start to the next period after the last received candle
                last_timestamp = df_batch.index[-1]
                current_start = last_timestamp + pd.Timedelta(milliseconds=1)

                # Avoid infinite loop if no new data is received
                if len(df_batch) < batch_size:
                    break

                # Small delay to avoid rate limiting
                time.sleep(0.2)

            if all_data.empty:
                raise ValueError("No data received from API after all batches")

            # Remove duplicates and sort by index
            all_data = all_data[~all_data.index.duplicated(keep='first')]
            all_data.sort_index(inplace=True)

            # Filter by the exact requested date range
            all_data = all_data.loc[start_date:end_date]

            # Save to CSV
            all_data.to_csv(filename)
            print(f"Saved {len(all_data)} records for {symbol} {interval} to {filename}")

            return all_data

        except Exception as e:
            print(f"Error fetching {symbol} {interval} data: {str(e)}")
            if not all_data.empty:
                print(f"Returning partial data ({len(all_data)} records)")
                return all_data
            return None

    def preprocess_data(self, data):
        """Preprocess the data for model input.
        返回: (close_scaled, volume_scaled) 两个数组的元组
        """
        close_scaled = self.scalers['close'].fit_transform(data['Close'].to_numpy().reshape(-1, 1)).reshape(-1)
        volume_scaled = self.scalers['volume'].fit_transform(data['Volume'].to_numpy().reshape(-1, 1)).reshape(-1)
        return close_scaled, volume_scaled

    def create_sequences(self, close_data, volume_data, input_window, output_window):
        """Create input-output sequences for training.
        输入: close_data和volume_data两个数组
        返回: 包含多个特征的序列列表
        """
        sequences = []
        L = len(close_data)
        for i in range(L - input_window - output_window + 1):
            # 创建包含多个特征的输入序列
            train_seq = np.column_stack([
                np.append(close_data[i:i + input_window], output_window * [0]),
                np.append(volume_data[i:i + input_window], output_window * [0])
            ])

            # 创建包含多个特征的目标序列
            train_label = np.column_stack([
                close_data[i:i + input_window + output_window],
                volume_data[i:i + input_window + output_window]
            ])

            sequences.append((train_seq, train_label))
        return sequences

    def load_or_download_data(self, symbol, interval, start_date=None, end_date=None, force_download=False):
        """Load data from file or download if not available."""
        filename = f"{symbol}_{interval}_data.joblib"
        if os.path.exists(filename) and not force_download:
            data = joblib.load(filename)
        else:
            data = self.get_historical_data(symbol, interval, start_date, end_date, force_download=True)
            if data is not None:
                joblib.dump(data, filename)
        return data

    def load_or_download_data_backtest(self, symbol, interval, start_date=None, end_date=None, force_download=False):
        """修复版本：确保正确加载指定日期范围的数据"""
        filename = f"{symbol}_{interval}_data_backtest.joblib"
        start_date = pd.to_datetime(start_date) if start_date else None
        end_date = pd.to_datetime(end_date) if end_date else None
        data = self.get_historical_data(symbol, interval, start_date, end_date, force_download=True)
        print(f'{start_date} -> {end_date} ->加载')
        return data

    def prepare_data(self, symbol, interval, input_window, output_window,
                     start_date=None, end_date=None, train_test_split=0.8):
        """Prepare data for training and testing."""
        data = self.load_or_download_data(symbol, interval, start_date, end_date)
        if data is None:
            raise ValueError("Failed to load data")

        close_scaled, volume_scaled = self.preprocess_data(data)
        sequences = self.create_sequences(close_scaled, volume_scaled, input_window, output_window)
        split_idx = int(len(sequences) * train_test_split)
        train_data = sequences[:split_idx]
        test_data = sequences[split_idx:]
        return train_data, test_data







# ---------------------- Package Metadata ----------------------

__all__ = [
    'DataLoader',
    'BinanceAPI'
]

__version__ = '0.2.3'