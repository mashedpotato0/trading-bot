import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import psutil
import gc
import warnings
import time
from datetime import datetime, timedelta
import asyncio
import concurrent.futures

warnings.filterwarnings('ignore')

def setup_pytorch_for_mx350():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        return torch.device('cpu')

device = setup_pytorch_for_mx350()

class DayTradingConfig:
    def __init__(self):
        self.POPULATION_SIZE = 50
        self.ELITE_SIZE = 20
        self.MEMORY_PER_BOT = 5000
        self.BATCH_SIZE = 64
        self.SEQUENCE_LENGTH = 30
        self.PARALLEL_WORKERS = 4
        self.GENERATIONS = 100
        self.EPISODES_PER_EVAL = 5
        self.MAX_CACHED_ASSETS = 20
        self.DATA_CHUNK_SIZE = 2000
        self.LEARNING_RATE = 0.0005
        self.GAMMA = 0.99
        self.TRAINING_WEEKS = 8
        self.TEST_WEEKS = 2
        self.REFRESH_INTERVAL = 60
        self.MIN_NEURONS = 32
        self.MAX_NEURONS = 256
        self.MIN_LAYERS = 2
        self.MAX_LAYERS = 6
        self.COMPLEXITY_THRESHOLD = 0.15

class RealTimeDataManager:
    def __init__(self, config):
        self.config = config
        self.training_cache = {}
        self.test_cache = {}
        self.last_update = None
        self.current_cached_assets = 0

    def load_training_data(self, tickers, weeks_back=8):
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks_back)

        self.training_cache = {}
        self.current_cached_assets = 0

        for ticker in tickers[:self.config.MAX_CACHED_ASSETS]:
            try:
                df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'), interval='1m', progress=False)
                if df.empty:
                    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'), interval='5m', progress=False)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                df = self._add_daytrading_indicators(df)
                df = df.tail(self.config.DATA_CHUNK_SIZE)
                self.training_cache[ticker] = df
                self.current_cached_assets += 1
            except Exception as e:
                print(f"Error loading training data for {ticker}: {e}")

        self.last_update = datetime.now()
        return self.training_cache

    def load_test_data(self, tickers, weeks_back=2):
        test_end = datetime.now() - timedelta(weeks=self.config.TRAINING_WEEKS)
        test_start = test_end - timedelta(weeks=weeks_back)

        self.test_cache = {}

        for ticker in tickers[:self.config.MAX_CACHED_ASSETS]:
            try:
                df = yf.download(ticker, start=test_start.strftime('%Y-%m-%d'),
                               end=test_end.strftime('%Y-%m-%d'), interval='1m', progress=False)
                if df.empty:
                    df = yf.download(ticker, start=test_start.strftime('%Y-%m-%d'),
                                   end=test_end.strftime('%Y-%m-%d'), interval='5m', progress=False)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                df = self._add_daytrading_indicators(df)
                self.test_cache[ticker] = df
            except Exception as e:
                print(f"Error loading test data for {ticker}: {e}")

        return self.test_cache

    def _add_daytrading_indicators(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()

        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()

        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()

        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()

        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        df['MACD'] = df['EMA_5'] - df['EMA_10']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        df['Bollinger_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['Bollinger_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
        df['Bollinger_Position'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])

        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(window=20).min()) / (df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())

        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1

        df['ATR'] = self._calculate_atr(df, 14)
        df['Williams_R'] = self._calculate_williams_r(df, 14)

        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        return df

    def _calculate_rsi(self, price, window):
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df, window):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()

    def _calculate_williams_r(self, df, window):
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        return -100 * (highest_high - df['Close']) / (highest_high - lowest_low)

    def needs_refresh(self):
        if self.last_update is None:
            return True
        return (datetime.now() - self.last_update).total_seconds() > self.config.REFRESH_INTERVAL

class DayTradingEnv(gym.Env):
    def __init__(self, df, init_balance=10000, transaction_cost=0.001, sequence_length=30):
        super(DayTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.init_balance = init_balance
        self.transaction_cost = transaction_cost
        self.sequence_length = sequence_length

        self.action_space = spaces.Discrete(11)

        market_features = len(df.columns)
        portfolio_features = 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, market_features + portfolio_features),
            dtype=np.float32
        )

        self.sequence_buffer = deque(maxlen=sequence_length)
        self.trade_log = []
        self.minute_returns = []
        self.reset()

    def reset(self):
        self.current_step = self.sequence_length
        self.balance = self.init_balance
        self.shares_held = 0.0
        self.total_value = self.init_balance
        self.max_net_worth = self.init_balance
        self.min_net_worth = self.init_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.daily_trades = 0
        self.trade_log = []
        self.minute_returns = []
        self.last_trade_price = 0

        self.sequence_buffer.clear()
        for i in range(self.sequence_length):
            obs = self._get_observation(i)
            self.sequence_buffer.append(obs)

        return np.array(list(self.sequence_buffer), dtype=np.float32)

    def _get_observation(self, step_idx):
        if step_idx >= len(self.df):
            step_idx = len(self.df) - 1

        market_data = self.df.iloc[step_idx].values.astype(np.float32)
        current_price = float(self.df.iloc[step_idx]['Close'])

        self.total_value = self.balance + self.shares_held * current_price

        portfolio_data = np.array([
            self.balance / self.init_balance,
            self.shares_held * current_price / self.init_balance,
            self.total_value / self.init_balance,
            (self.total_value - self.init_balance) / self.init_balance,
            self.consecutive_losses / 10.0,
            self.daily_trades / 50.0
        ], dtype=np.float32)

        return np.concatenate([market_data, portfolio_data])

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return np.array(list(self.sequence_buffer)), 0, True, self._get_info()

        current_price = float(self.df.iloc[self.current_step]['Close'])
        prev_value = self.total_value
        trade_info = None

        action_mapping = {
            0: 0.0,    # Hold
            1: -1.0,   # Sell all
            2: -0.5,   # Sell half
            3: -0.25,  # Sell quarter
            4: -0.1,   # Sell 10%
            5: 0.1,    # Buy 10% of balance
            6: 0.25,   # Buy 25% of balance
            7: 0.5,    # Buy 50% of balance
            8: 0.75,   # Buy 75% of balance
            9: 1.0,    # Buy all balance
            10: 1.5    # Buy with 50% margin
        }

        trade_fraction = action_mapping[action]

        if trade_fraction < 0 and self.shares_held > 0:
            shares_to_sell = abs(trade_fraction) * self.shares_held
            if trade_fraction == -1.0:
                shares_to_sell = self.shares_held

            sell_value = shares_to_sell * current_price * (1 - self.transaction_cost)

            trade_info = {
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'amount': sell_value,
                'step': self.current_step
            }

            self.balance += sell_value
            self.shares_held -= shares_to_sell
            self.total_trades += 1
            self.daily_trades += 1

            if self.last_trade_price > 0:
                if current_price > self.last_trade_price:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

        elif trade_fraction > 0 and self.balance > 0:
            available_balance = self.balance
            if trade_fraction > 1.0:
                available_balance *= trade_fraction

            investment_amount = available_balance * min(trade_fraction, 1.0)
            shares_to_buy = investment_amount / (current_price * (1 + self.transaction_cost))

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)

                if cost <= self.balance or trade_fraction > 1.0:
                    trade_info = {
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'amount': cost,
                        'step': self.current_step
                    }

                    self.shares_held += shares_to_buy
                    self.balance -= min(cost, self.balance)
                    self.total_trades += 1
                    self.daily_trades += 1
                    self.last_trade_price = current_price

        if trade_info:
            self.trade_log.append(trade_info)

        self.current_step += 1
        obs = self._get_observation(self.current_step)
        self.sequence_buffer.append(obs)

        reward = self._calculate_daytrading_reward(prev_value)

        self.max_net_worth = max(self.max_net_worth, self.total_value)
        self.min_net_worth = min(self.min_net_worth, self.total_value)

        minute_return = (self.total_value - prev_value) / prev_value if prev_value > 0 else 0
        self.minute_returns.append(minute_return)

        done = self.current_step >= len(self.df) - 1

        return np.array(list(self.sequence_buffer)), reward, done, self._get_info()

    def _calculate_daytrading_reward(self, prev_value):
        if prev_value <= 0:
            return -100

        value_change = self.total_value - prev_value
        portfolio_return = value_change / prev_value

        base_reward = 0
        if portfolio_return > 0:
            base_reward = portfolio_return * 500
            if portfolio_return > 0.01:
                base_reward *= 2
        else:
            penalty_multiplier = 3
            if portfolio_return < -0.01:
                penalty_multiplier = 5
            if portfolio_return < -0.02:
                penalty_multiplier = 10
            base_reward = portfolio_return * penalty_multiplier * 500

        drawdown = (self.max_net_worth - self.total_value) / self.max_net_worth if self.max_net_worth > 0 else 0
        drawdown_penalty = -drawdown * 200

        consecutive_loss_penalty = -self.consecutive_losses * 50

        overtrade_penalty = 0
        if self.daily_trades > 20:
            overtrade_penalty = -(self.daily_trades - 20) * 10

        win_rate = self.winning_trades / max(self.total_trades, 1)
        win_rate_bonus = (win_rate - 0.5) * 100 if self.total_trades > 5 else 0

        volatility_penalty = 0
        if len(self.minute_returns) > 10:
            volatility = np.std(self.minute_returns[-10:])
            if volatility > 0.02:
                volatility_penalty = -volatility * 1000

        balance_penalty = 0
        total_capital = self.balance + self.shares_held * self.df.iloc[self.current_step]['Close']
        if self.balance / total_capital > 0.9:
            balance_penalty = -20

        total_reward = (base_reward + drawdown_penalty + consecutive_loss_penalty +
                       overtrade_penalty + win_rate_bonus + volatility_penalty + balance_penalty)

        return total_reward

    def _get_info(self):
        return {
            'total_value': self.total_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'portfolio_return': (self.total_value - self.init_balance) / self.init_balance,
            'max_drawdown': (self.max_net_worth - self.min_net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0,
            'daily_trades': self.daily_trades,
            'trade_log': self.trade_log
        }

class DynamicTradingNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sizes, num_layers, dropout_rate=0.2):
        super(DynamicTradingNetwork, self).__init__()

        sequence_length, feature_size = input_shape
        self.feature_size = feature_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers

        self.lstm_layers = nn.ModuleList()
        current_input_size = feature_size

        for i in range(num_layers):
            hidden_size = hidden_sizes[i] if i < len(hidden_sizes) else hidden_sizes[-1]
            self.lstm_layers.append(
                nn.LSTM(current_input_size, hidden_size, batch_first=True, dropout=dropout_rate if i < num_layers-1 else 0)
            )
            current_input_size = hidden_size

        final_hidden_size = hidden_sizes[-1] if hidden_sizes else 64

        self.attention = nn.MultiheadAttention(final_hidden_size, num_heads=4, dropout=dropout_rate)

        self.fc_layers = nn.ModuleList()
        fc_sizes = [final_hidden_size, final_hidden_size//2, final_hidden_size//4, 11]

        for i in range(len(fc_sizes)-1):
            self.fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))
            if i < len(fc_sizes)-2:
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_rate))

        self.layer_norm = nn.LayerNorm(final_hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        lstm_out = x
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)

        lstm_out = self.layer_norm(lstm_out)

        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)

        final_out = lstm_out + attn_out
        final_out = final_out[:, -1, :]

        for layer in self.fc_layers:
            final_out = layer(final_out)

        return F.softmax(final_out, dim=1)

class EvolutionaryTradingBot:
    def __init__(self, input_shape, config, network_config=None):
        self.input_shape = input_shape
        self.config = config
        self.device = device

        if network_config is None:
            self.hidden_sizes = [random.randint(config.MIN_NEURONS, config.MAX_NEURONS)
                               for _ in range(random.randint(config.MIN_LAYERS, config.MAX_LAYERS))]
            self.num_layers = len(self.hidden_sizes)
            self.dropout_rate = random.uniform(0.1, 0.3)
        else:
            self.hidden_sizes = network_config['hidden_sizes']
            self.num_layers = network_config['num_layers']
            self.dropout_rate = network_config['dropout_rate']

        self.network = DynamicTradingNetwork(input_shape, self.hidden_sizes, self.num_layers, self.dropout_rate).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = deque(maxlen=config.MEMORY_PER_BOT)

        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.fitness_history = []
        self.generation = 0
        self.best_fitness = float('-inf')
        self.complexity_score = self._calculate_complexity()

    def _calculate_complexity(self):
        total_params = sum(p.numel() for p in self.network.parameters())
        return total_params / 1000000

    def act(self, state, deterministic=False):
        if not deterministic and np.random.random() <= self.epsilon:
            return np.random.randint(0, 11)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.network(state_tensor)
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
            else:
                action = torch.multinomial(action_probs, 1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = min(self.config.BATCH_SIZE, len(self.memory))
        if len(self.memory) < batch_size:
            return 0

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.network(states)

        with torch.no_grad():
            next_q_values = self.network(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]

        target_q_values = rewards + (self.config.GAMMA * max_next_q * ~dones)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        policy_loss = self.criterion(current_q_values, actions)
        value_loss = F.mse_loss(current_q, target_q_values)
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss.item()

    def evolve_architecture(self, parent_fitness, elite_fitness_threshold):
        should_complexify = (parent_fitness > elite_fitness_threshold and
                           random.random() < self.config.COMPLEXITY_THRESHOLD)

        if should_complexify:
            if random.random() < 0.5 and self.num_layers < self.config.MAX_LAYERS:
                new_size = random.randint(self.config.MIN_NEURONS, self.config.MAX_NEURONS)
                insert_pos = random.randint(0, len(self.hidden_sizes))
                self.hidden_sizes.insert(insert_pos, new_size)
                self.num_layers += 1
            else:
                for i in range(len(self.hidden_sizes)):
                    if random.random() < 0.3:
                        increase = random.randint(8, 32)
                        self.hidden_sizes[i] = min(self.hidden_sizes[i] + increase, self.config.MAX_NEURONS)
        else:
            if random.random() < 0.2 and self.num_layers > self.config.MIN_LAYERS:
                remove_pos = random.randint(0, len(self.hidden_sizes)-1)
                self.hidden_sizes.pop(remove_pos)
                self.num_layers -= 1
            else:
                for i in range(len(self.hidden_sizes)):
                    if random.random() < 0.2:
                        decrease = random.randint(4, 16)
                        self.hidden_sizes[i] = max(self.hidden_sizes[i] - decrease, self.config.MIN_NEURONS)

    def clone_and_mutate(self, mutation_rate=0.02, elite_fitness_threshold=0.1):
        network_config = {
            'hidden_sizes': self.hidden_sizes.copy(),
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate
        }

        child = EvolutionaryTradingBot(self.input_shape, self.config, network_config)

        if self.best_fitness > elite_fitness_threshold:
            child.evolve_architecture(self.best_fitness, elite_fitness_threshold)

        child.network = DynamicTradingNetwork(self.input_shape, child.hidden_sizes,
                                            child.num_layers, child.dropout_rate).to(self.device)
        child.optimizer = optim.AdamW(child.network.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-4)

        with torch.no_grad():
            for child_param, parent_param in zip(child.network.parameters(), self.network.parameters()):
                if child_param.shape == parent_param.shape:
                    if np.random.random() < 0.4:
                        mutation_strength = np.random.choice([0.01, 0.02, 0.05, 0.1], p=[0.4, 0.3, 0.2, 0.1])
                        noise = torch.randn_like(parent_param) * mutation_strength
                        child_param.data = parent_param.data + noise
                    else:
                        child_param.data = parent_param.data.clone()
                else:
                    nn.init.xavier_uniform_(child_param)

        child.epsilon = self.epsilon * np.random.uniform(0.9, 1.1)
        child.generation = self.generation + 1
        child.complexity_score = child._calculate_complexity()

        return child

def evaluate_bot_async(bot, data_dict, episodes=3):
    results = []
    assets = list(data_dict.keys())
    selected_assets = random.sample(assets, min(5, len(assets)))

    for asset in selected_assets:
        data = data_dict[asset]
        env = DayTradingEnv(data, init_balance=10000, sequence_length=bot.config.SEQUENCE_LENGTH)

        episode_results = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = bot.act(state, deterministic=False)
                next_state, reward, done, info = env.step(action)
                bot.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(bot.memory) > bot.config.BATCH_SIZE:
                    bot.train_batch()

            episode_results.append({
                'reward': episode_reward,
                'final_value': info['total_value'],
                'return': info['portfolio_return'],
                'trades': info['total_trades'],
                'win_rate': info['win_rate'],
                'max_drawdown': info['max_drawdown']
            })

        avg_return = np.mean([r['return'] for r in episode_results])
        avg_reward = np.mean([r['reward'] for r in episode_results])
        avg_trades = np.mean([r['trades'] for r in episode_results])
        avg_win_rate = np.mean([r['win_rate'] for r in episode_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in episode_results])

        results.append({
            'asset': asset,
            'avg_return': avg_return,
            'avg_reward': avg_reward,
            'avg_trades': avg_trades,
            'avg_win_rate': avg_win_rate,
            'avg_drawdown': avg_drawdown
        })

    overall_return = np.mean([r['avg_return'] for r in results])
    overall_reward = np.mean([r['avg_reward'] for r in results])
    overall_win_rate = np.mean([r['avg_win_rate'] for r in results])
    overall_drawdown = np.mean([r['avg_drawdown'] for r in results])

    sharpe_ratio = overall_return / (np.std([r['avg_return'] for r in results]) + 1e-8)

    complexity_penalty = bot.complexity_score * 0.1
    fitness = (overall_return * 100 + sharpe_ratio * 50 + overall_win_rate * 30 -
               overall_drawdown * 50 - complexity_penalty)

    bot.best_fitness = max(bot.best_fitness, fitness)
    bot.fitness_history.append(fitness)

    return fitness, results

class EvolutionaryTradingSystem:
    def __init__(self, config):
        self.config = config
        self.data_manager = RealTimeDataManager(config)
        self.population = []
        self.generation = 0
        self.best_bot = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.generation_stats = []

    def initialize_population(self, input_shape):
        self.population = []
        for i in range(self.config.POPULATION_SIZE):
            bot = EvolutionaryTradingBot(input_shape, self.config)
            self.population.append(bot)
        print(f"Initialized population of {len(self.population)} bots")

    def evaluate_population(self, training_data):
        print(f"Evaluating generation {self.generation}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.PARALLEL_WORKERS) as executor:
            futures = []
            for bot in self.population:
                future = executor.submit(evaluate_bot_async, bot, training_data, self.config.EPISODES_PER_EVAL)
                futures.append(future)

            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    fitness, details = future.result()
                    results.append((i, fitness, details))
                    print(f"Bot {i+1}/{len(self.population)} evaluated: fitness = {fitness:.2f}")
                except Exception as e:
                    print(f"Error evaluating bot {i}: {e}")
                    results.append((i, float('-inf'), []))

        # Sort by fitness
        results.sort(key=lambda x: x[1], reverse=True)

        # Update fitness tracking
        current_fitnesses = [r[1] for r in results]
        self.fitness_history.extend(current_fitnesses)

        # Update best bot
        if results[0][1] > self.best_fitness:
            self.best_fitness = results[0][1]
            best_bot_idx = results[0][0]
            self.best_bot = self.population[best_bot_idx]
            print(f"New best bot found! Fitness: {self.best_fitness:.2f}")

        # Generation statistics
        gen_stats = {
            'generation': self.generation,
            'best_fitness': results[0][1],
            'avg_fitness': np.mean(current_fitnesses),
            'std_fitness': np.std(current_fitnesses),
            'worst_fitness': results[-1][1]
        }
        self.generation_stats.append(gen_stats)

        return results

    def evolve_population(self, evaluation_results):
        # Select elite bots
        elite_size = self.config.ELITE_SIZE
        elite_indices = [r[0] for r in evaluation_results[:elite_size]]
        elite_bots = [self.population[i] for i in elite_indices]

        # Calculate fitness threshold for elite performance
        elite_fitness_threshold = np.mean([r[1] for r in evaluation_results[:elite_size]])

        # Create new population
        new_population = []

        # Keep elite bots
        for bot in elite_bots:
            new_population.append(bot)

        # Generate offspring from elite bots
        while len(new_population) < self.config.POPULATION_SIZE:
            parent = random.choice(elite_bots)
            child = parent.clone_and_mutate(elite_fitness_threshold=elite_fitness_threshold)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Memory cleanup
        if self.generation % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_evolution(self, tickers):
        print("Loading training data...")
        training_data = self.data_manager.load_training_data(tickers)

        if not training_data:
            print("No training data available!")
            return

        # Get input shape from first available dataset
        sample_data = list(training_data.values())[0]
        input_shape = (self.config.SEQUENCE_LENGTH, len(sample_data.columns) + 6)

        print("Initializing population...")
        self.initialize_population(input_shape)

        for generation in range(self.config.GENERATIONS):
            print(f"\n=== Generation {generation + 1}/{self.config.GENERATIONS} ===")

            # Refresh data periodically
            if self.data_manager.needs_refresh():
                print("Refreshing training data...")
                training_data = self.data_manager.load_training_data(tickers)

            # Evaluate population
            evaluation_results = self.evaluate_population(training_data)

            # Print generation summary
            best_fitness = evaluation_results[0][1]
            avg_fitness = np.mean([r[1] for r in evaluation_results])
            print(f"Generation {generation + 1} - Best: {best_fitness:.2f}, Avg: {avg_fitness:.2f}")

            # Evolve population
            if generation < self.config.GENERATIONS - 1:
                self.evolve_population(evaluation_results)

        print(f"\nEvolution completed! Best overall fitness: {self.best_fitness:.2f}")
        return self.best_bot

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Evolutionary Day Trading Bot")
        self.root.geometry("1200x800")

        self.config = DayTradingConfig()
        self.trading_system = EvolutionaryTradingSystem(self.config)
        self.is_running = False

        self.setup_gui()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Tickers input
        ttk.Label(config_frame, text="Tickers (comma-separated):").grid(row=0, column=0, sticky=tk.W)
        self.tickers_var = tk.StringVar(value="AAPL,GOOGL,MSFT,TSLA,NVDA,AMD,SPY,QQQ")
        ttk.Entry(config_frame, textvariable=self.tickers_var, width=60).grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Parameters
        ttk.Label(config_frame, text="Population Size:").grid(row=1, column=0, sticky=tk.W)
        self.pop_size_var = tk.IntVar(value=self.config.POPULATION_SIZE)
        ttk.Spinbox(config_frame, from_=20, to=200, textvariable=self.pop_size_var, width=10).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(config_frame, text="Generations:").grid(row=2, column=0, sticky=tk.W)
        self.generations_var = tk.IntVar(value=self.config.GENERATIONS)
        ttk.Spinbox(config_frame, from_=10, to=500, textvariable=self.generations_var, width=10).grid(row=2, column=1, sticky=tk.W)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Evolution", command=self.start_evolution)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_evolution, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Progress and status
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.status_var = tk.StringVar(value="Ready to start evolution")
        ttk.Label(status_frame, textvariable=self.status_var).pack()

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

    def start_evolution(self):
        if self.is_running:
            return

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Update config from GUI
        self.config.POPULATION_SIZE = self.pop_size_var.get()
        self.config.GENERATIONS = self.generations_var.get()

        # Get tickers
        tickers = [t.strip().upper() for t in self.tickers_var.get().split(',')]

        # Start evolution in separate thread
        self.evolution_thread = threading.Thread(target=self.run_evolution_thread, args=(tickers,))
        self.evolution_thread.start()

    def stop_evolution(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopping evolution...")

    def run_evolution_thread(self, tickers):
        try:
            self.status_var.set("Loading data and initializing...")
            best_bot = self.trading_system.run_evolution(tickers)

            if best_bot:
                self.status_var.set(f"Evolution completed! Best fitness: {self.trading_system.best_fitness:.2f}")
                self.update_results_display()
            else:
                self.status_var.set("Evolution failed - no data available")

        except Exception as e:
            self.status_var.set(f"Error during evolution: {str(e)}")
        finally:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def update_results_display(self):
        if not self.trading_system.generation_stats:
            return

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        generations = [s['generation'] for s in self.trading_system.generation_stats]
        best_fitness = [s['best_fitness'] for s in self.trading_system.generation_stats]
        avg_fitness = [s['avg_fitness'] for s in self.trading_system.generation_stats]

        # Plot fitness evolution
        self.ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        self.ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', alpha=0.7)
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.set_title('Fitness Evolution')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Plot fitness distribution for last generation
        if len(self.trading_system.fitness_history) > 0:
            recent_fitness = self.trading_system.fitness_history[-self.config.POPULATION_SIZE:]
            self.ax2.hist(recent_fitness, bins=20, alpha=0.7, edgecolor='black')
            self.ax2.set_xlabel('Fitness')
            self.ax2.set_ylabel('Count')
            self.ax2.set_title('Current Generation Fitness Distribution')
            self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
