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

warnings.filterwarnings('ignore')

import os
np.random.seed(int(time.time() * 1000) % 2**32)
random.seed(int(time.time() * 1000) % 2**32)
all_csvs_train = [f for f in os.listdir("./stock_data/test") if f.endswith(".csv")]
all_csvs_test = [f for f in os.listdir("./stock_data/test") if f.endswith(".csv")]

def load_ticker_universe(path='./tickers.csv'):
    df = pd.read_csv(path)
    return df['Symbol'].dropna().unique().tolist()

def get_random_tickers(count=13, folder="./stock_data/test"):
    all_csvs = [f for f in os.listdir(folder) if f.endswith(".csv")]
    available_tickers = [f[:-4] for f in all_csvs]  # skibiddy del the extension

    if len(available_tickers) < count:
        raise ValueError(f"Not enough tickers in {folder}. Needed: {count}, found: {len(available_tickers)}")

    return random.sample(available_tickers, count)




def setup_pytorch_for_mx350():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        return torch.device('cpu')

device = setup_pytorch_for_mx350()

class OptimizedRAMConfig:
    def __init__(self):
        self.POPULATION_SIZE = 50
        self.ELITE_SIZE = 15
        self.MEMORY_PER_BOT = 4000
        self.BATCH_SIZE = 32
        self.SEQUENCE_LENGTH = 45
        self.PARALLEL_WORKERS = 2
        self.GENERATIONS = 50
        self.EPISODES_PER_EVAL = 3
        self.MAX_CACHED_ASSETS = 12
        self.DATA_CHUNK_SIZE = 800
        self.LEARNING_RATE = 0.005
        self.GAMMA = 0.95 #random bulshit coz my pc anit pcINg

class LightweightDataManager:
    def __init__(self, ram_config):
        self.ram_config = ram_config
        self.data_cache = {}
        self.current_cached_assets = 0

    def load_essential_assets(self, tickers=None, start="2021-01-01", end="2024-01-01", use_csv=False, csv_folder="./stock_data"):
        self.data_cache = {}
        self.current_cached_assets = 0

        if use_csv:
            return self.load_from_csvs(csv_folder, tickers=tickers)

        if tickers is None:
            tickers = get_random_tickers(12)

        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                df = self._add_essential_indicators(df)
                df = df.tail(self.ram_config.DATA_CHUNK_SIZE)
                self.data_cache[ticker] = df
                self.current_cached_assets += 1
            except Exception as e:
                print(f"error loading {ticker}: {e}")
        return self.data_cache

    def load_from_csvs(self, folder_path="./stock_data", tickers=None):
        self.data_cache = {}
        self.current_cached_assets = 0

        if not os.path.exists(folder_path):
            print(f"folder not found: {folder_path}")
            return self.data_cache

        all_files = os.listdir(folder_path)
        selected_files = []

        if tickers is not None:
            selected_files = [f"{t}.csv" for t in tickers if f"{t}.csv" in all_files]
        else:
            selected_files = [f for f in all_files if f.endswith(".csv")]

        for filename in selected_files:
            ticker = filename[:-4]
            try:
                df = pd.read_csv(os.path.join(folder_path, filename), index_col=0, parse_dates=True)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = self._add_essential_indicators(df)
                df = df.tail(self.ram_config.DATA_CHUNK_SIZE)
                self.data_cache[ticker] = df
                self.current_cached_assets += 1
                if self.current_cached_assets == 12:
                    break
            except Exception as e:
                print(f"failed to load {filename}: {e}")

        return self.data_cache



    def test_assets(self, tickers=None, start="2020-01-01", end="2024-01-01"):
        self.data_cache = {}
        self.current_cached_assets = 0

        if tickers is None:
            tickers = load_from_csvs(12)

        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                df = self._add_essential_indicators(df)
                df = df.tail(self.ram_config.DATA_CHUNK_SIZE)
                self.data_cache[ticker] = df
                self.current_cached_assets += 1
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
        return self.data_cache

    def _add_essential_indicators(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=15).mean()
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        return df

    def _calculate_rsi(self, price, window):
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class LightweightTradingEnv(gym.Env):
    def __init__(self, df, init_balance=1000, transaction_cost=0.001, sequence_length=20):
        super(LightweightTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.init_balance = init_balance
        self.transaction_cost = transaction_cost
        self.sequence_length = sequence_length
        self.action_space = spaces.Discrete(3)
        market_features = len(df.columns)
        portfolio_features = 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, market_features + portfolio_features),
            dtype=np.float32
        )
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.trade_log = []
        self.reset()

    def reset(self):
        self.current_step = self.sequence_length
        self.balance = self.init_balance
        self.shares_held = 0
        self.total_value = self.init_balance
        self.max_net_worth = self.init_balance
        self.total_trades = 0
        self.trade_log = []
        self.sequence_buffer.clear()
        for i in range(self.sequence_length):
            obs = self._get_observation(i)
            self.sequence_buffer.append(obs)
        return np.array(list(self.sequence_buffer), dtype=np.float32)
    def load_from_csvs(self, folder_path="./stock_data"):
        self.data_cache = {}
        self.current_cached_assets = 0
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} not found.")
            return self.data_cache

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                ticker = filename[:-4]
                try:
                    df = pd.read_csv(os.path.join(folder_path, filename), index_col=0, parse_dates=True)
                    df = self._add_essential_indicators(df)
                    df = df.tail(self.ram_config.DATA_CHUNK_SIZE)
                    self.data_cache[ticker] = df
                    self.current_cached_assets += 1
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
        return self.data_cache


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
            (self.total_value - self.init_balance) / self.init_balance
        ], dtype=np.float32)
        return np.concatenate([market_data, portfolio_data])

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return np.array(list(self.sequence_buffer)), 0, True, self._get_info()
        current_price = float(self.df.iloc[self.current_step]['Close'])
        prev_value = self.total_value
        trade_info = None

        if action == 0 and self.shares_held > 0:
            sell_amount = self.shares_held * current_price * (1 - self.transaction_cost)
            trade_info = {
                'action': 'SELL',
                'shares': self.shares_held,
                'price': current_price,
                'amount': sell_amount,
                'step': self.current_step
            }
            self.balance += sell_amount
            self.shares_held = 0
            self.total_trades += 1
        elif action == 2 and self.balance > 0:
            shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                trade_info = {
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'amount': cost,
                    'step': self.current_step
                }
                self.shares_held += shares_to_buy
                self.balance -= cost
                self.total_trades += 1

        if trade_info:
            self.trade_log.append(trade_info)

        self.current_step += 1
        obs = self._get_observation(self.current_step)
        self.sequence_buffer.append(obs)
        reward = self._calculate_reward(prev_value)
        self.max_net_worth = max(self.max_net_worth, self.total_value)
        done = self.current_step >= len(self.df) - 1
        return np.array(list(self.sequence_buffer)), reward, done, self._get_info()

    def _calculate_reward(self, prev_value):
        if prev_value <= 0:
            return -10

        portfolio_return = (self.total_value - prev_value) / prev_value
        reward = portfolio_return * 100

        if self.total_trades == 0:
            reward -= np.log1p(self.total_trades+2) * 20

        if self.total_trades > 1:
            reward += np.log1p(self.total_trades) * 10

        if self.max_net_worth > 0:
            drawdown = (self.max_net_worth - self.total_value) / self.max_net_worth
            reward -= drawdown * 5

        return reward



    def _get_info(self):
        return {
            'total_value': self.total_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'portfolio_return': (self.total_value - self.init_balance) / self.init_balance,
            'trade_log': self.trade_log
        }

class TradingNetwork(nn.Module):
    def __init__(self, input_shape, layer_config=None):
        super(TradingNetwork, self).__init__()
        self.input_shape = input_shape
        self.layer_config = layer_config or [128, 64, 32]
        self.layers = nn.ModuleList()

        input_size = input_shape[1]
        for hidden_size in self.layer_config:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        self.output_layer = nn.Linear(input_size, 3)

    def forward(self, x):
        x = x[:, -1, :]  # loop the shit and make connecting layers
        for layer in self.layers:
            x = F.relu(layer(x))
        return F.softmax(self.output_layer(x), dim=1)



class PyTorchTradingBot:
    def __init__(self, input_shape, ram_config, learning_rate=0.001, layer_config=None):
        self.input_shape = input_shape
        self.ram_config = ram_config
        self.device = device
        self.network = TradingNetwork(input_shape, layer_config=layer_config).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = deque(maxlen=ram_config.MEMORY_PER_BOT)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.fitness_history = []
        self.generation = 0
        self.best_fitness = float('-inf')

    def save_model(self, filepath="best_trading_model.pth"):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'generation': self.generation,
            'layer_config': self.network.layer_config
        }, filepath)
    def add_essential_indicators(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=15).mean()

        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        return df

    def calculate_rsi(self, price, window):
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


    def meta_train(self, tickers, episodes=2, inner_steps=3, alpha=0.01):
        meta_grads = [torch.zeros_like(p) for p in self.network.parameters()]

        for ticker in tickers:
            try:
                df = pd.read_csv(f"./stock_data/train_fixed/{ticker}.csv", index_col=0, parse_dates=True)

                # wel i generated the fixed part coz well that was the only way it worked
                expected_features = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Returns', 'SMA_10', 'SMA_20', 'EMA_12',
                    'Volatility_10', 'RSI', 'Price_Change', 'Volume_Ratio'
                ]
                df = df[expected_features]
                df = df.apply(pd.to_numeric, errors='coerce')
                df.dropna(inplace=True)

                input_dim = df.shape[1] + 4  # +4 for portfolio info added in env
                expected_dim = self.input_shape[1]
                if input_dim != expected_dim:
                    print(f"skipping {ticker}: expected {expected_dim} features, got {input_dim}")
                    continue

            except Exception as e:
                print(f"skipping {ticker}: {e}")
                continue

            sequence_length = self.ram_config.SEQUENCE_LENGTH
            input_shape = (sequence_length, input_dim)
            env = LightweightTradingEnv(df, sequence_length=sequence_length)
            adapted_bot = PyTorchTradingBot(input_shape, self.ram_config, layer_config=self.network.layer_config)
            adapted_bot.network.load_state_dict(self.network.state_dict())

            for _ in range(inner_steps):
                done = False
                state = env.reset()
                while not done:
                    action = adapted_bot.act(state)
                    next_state, reward, done, _ = env.step(action)
                    adapted_bot.remember(state, action, reward, next_state, done)
                    state = next_state
                adapted_bot.train_batch()

            adapted_bot.optimizer.zero_grad()
            adapted_bot.train_batch()
            for meta_g, adapted_p in zip(meta_grads, adapted_bot.network.parameters()):
                if adapted_p.grad is not None:
                    meta_g += adapted_p.grad

        for p, g in zip(self.network.parameters(), meta_grads):
            p.data -= alpha * g / len(tickers)



    def clone(self):
        clone = PyTorchTradingBot(self.input_shape, self.ram_config, layer_config=self.network.layer_config)
        clone.network.load_state_dict(self.network.state_dict())
        return clone


    def load_model(self, filepath="best_trading_model.pth"):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            layer_config = checkpoint.get('layer_config', [128, 64, 32] )
            self.network = TradingNetwork(self.input_shape, layer_config=layer_config).to(self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.generation = checkpoint.get('generation', 0)
            print(f"loaded model from {filepath}")
            return True
        else:
            print(f"no model found at {filepath}")
            return False


    def act(self, state, deterministic=False):
        if not deterministic and np.random.random() <= self.epsilon:
            return np.random.randint(0, 3)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = min(self.ram_config.BATCH_SIZE, len(self.memory))
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
        target_q_values = rewards + (self.ram_config.GAMMA * max_next_q * ~dones)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q_values, actions) + F.mse_loss(current_q, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    def clone_and_mutate(self, mutation_rate=0.02, train_steps=3, meta_tickers=None):
        fixed_config = [128, 64, 32]
        child_input_shape = self.input_shape
        child = PyTorchTradingBot(child_input_shape, self.ram_config, layer_config=fixed_config)

        # Copy parameters with noise
        with torch.no_grad():
            for cp, pp in zip(child.network.parameters(), self.network.parameters()):
                if cp.shape == pp.shape:
                    noise = torch.randn_like(pp) * mutation_rate
                    cp.data = pp.data + noise

        child.epsilon = max(self.epsilon * np.random.uniform(0.95, 1.05), 0.01)
        child.generation = self.generation + 1
        child.memory = self.memory.copy()

        # please dont change this part this shit is inhumane and a big bully
        if meta_tickers:
            child.meta_train(meta_tickers, inner_steps=train_steps)

        return child



def evaluate_bot_threaded(bot, data_dict, episodes=2):
    results = []
    total_rewards = []
    selected_assets = list(data_dict.keys())  # idk just ls coz you aint need any random bullshit

    for asset in selected_assets:
        data = data_dict[asset]
        env = LightweightTradingEnv(data, sequence_length=bot.ram_config.SEQUENCE_LENGTH)
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = bot.act(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            bot.remember(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        results.append(info['portfolio_return'])

        for _ in range(3):
            bot.train_batch()

    mean_return = np.mean(results)
    mean_reward = np.mean(total_rewards)

    fitness = 0.7 * mean_return * 100  # add the rewads if you want to but pls dont
    best_asset_index = np.argmax(results)
    best_asset = selected_assets[best_asset_index]
    best_asset_return = results[best_asset_index]

    return bot, fitness, mean_return, total_rewards, best_asset, best_asset_return




class TradingBotUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("enhanced PyTorch Trading Bot Visualizer")
        self.root.geometry("1280x720")

        self.ram_config = OptimizedRAMConfig()
        self.data_manager = LightweightDataManager(self.ram_config)
        self.evolution_running = False
        self.evolution_thread = None
        self.best_bots = []
        self.historical_best_bot = None
        self.historical_best_fitness = float('-inf')
        self.chunk_best_bots = []
        self.chunk_start_generation = 0
        self.chunk_best_fitness = float('-inf')
        self.chunk_best_bot = None

        self.evolution_history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_return': [],
            'generation': []
        }

        self.setup_ui()

    def save_chunk_bots(self):
        if not self.chunk_best_bots:
            messagebox.showwarning("No Bots", "No chunk bots available to save.")
            return

        os.makedirs("saved_chunk_bots", exist_ok=True)

        for start, end, bot in self.chunk_best_bots:
            fname = f"saved_chunk_bots/bot_gen_{start}_to_{end}.pt"
            torch.save(bot.network.state_dict(), fname)
            self.update_stats(f"saved bot from gen {start}–{end} to {fname}")

    def visualize_best_network(self):
        if not self.historical_best_bot:
            self.update_stats("no best bot available.")
            return

        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter as tk
        import tkinter.ttk as ttk

        # diddy  this badboy up
        net = self.historical_best_bot.network
        input_size = net.input_shape[1]
        layer_sizes = [input_size] + net.layer_config + [3]

        # well aint gonna show all of em so
        max_layers = 8
        max_neurons = 10
        if len(layer_sizes) > max_layers:
            layer_sizes = layer_sizes[:max_layers]

        for i in range(len(layer_sizes)):
            layer_sizes[i] = min(layer_sizes[i], max_neurons)

        G = nx.DiGraph()
        positions = {}
        node_labels = {}
        layer_spacing = 2
        neuron_spacing = 1.5
        WEIGHTS_THRESHOLD = 0.01

        node_id = 0
        node_map = []
        edge_weights = []

        for l, size in enumerate(layer_sizes):
            layer_nodes = []
            for n in range(size):
                node_name = f"n{node_id}"
                G.add_node(node_name)
                positions[node_name] = (l * layer_spacing, -n * neuron_spacing)
                node_labels[node_name] = "o"
                layer_nodes.append(node_name)
                node_id += 1
            node_map.append(layer_nodes)

        for i in range(len(layer_sizes) - 1):
            layer1 = node_map[i]
            layer2 = node_map[i + 1]

            try:
                weights = net.layers[i].weight.detach().cpu().numpy() if i < len(net.layers) else net.output_layer.weight.detach().cpu().numpy()
            except:
                weights = None
            MAX = np.abs(weights).max() + 1

            for src_idx, src in enumerate(layer1):
                for dst_idx, dst in enumerate(layer2):
                    color = 'gray'
                    weight = weights[dst_idx][src_idx]/MAX
                    if (abs(weight)>WEIGHTS_THRESHOLD):
                        try:
                            color = 'red' if weight > 0 else 'green'
                            width = abs(weight) * 5
                            G.add_edge(src, dst, color=color, weight=width)
                        except:
                            pass

        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]


        # new window , i  have done enough overlaping shit
        win = tk.Toplevel(self.root)
        win.title("Neural Network Structure")
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(G, pos=positions, ax=ax, with_labels=False,
            node_color='skyblue', edge_color=edge_colors,
            node_size=500, width=edge_widths)
        for node, (x, y) in positions.items():
            ax.text(x, y, "o", ha='center', va='center', fontsize=10, weight='bold')

        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        # add scrollbar coz why not
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.get_tk_widget().yview)
        scrollbar.pack(side="right", fill="y")
        canvas.get_tk_widget().configure(yscrollcommand=scrollbar.set)

        self.update_stats("Visualized best network.")



    def setup_ui(self):

        """next fixes are the ui im just done with this shit.
        if i visited this in another 10 years """
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        control_frame.columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        ttk.Label(control_frame, text="Tickers:").grid(row=0, column=0, sticky=tk.W)
        self.ticker_entry = ttk.Entry(control_frame, width=40)
        self.ticker_entry.insert(0, "SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,BRK-B,JNJ,V,JPM,PG,UNH")
        self.ticker_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5)

        ttk.Label(control_frame, text="Population:").grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        self.pop_var = tk.StringVar(value="50")
        ttk.Entry(control_frame, textvariable=self.pop_var, width=8).grid(row=0, column=4, padx=5)

        ttk.Label(control_frame, text="Generations:").grid(row=0, column=5, sticky=tk.W, padx=(10, 0))
        self.gen_var = tk.StringVar(value="1000")
        ttk.Entry(control_frame, textvariable=self.gen_var, width=8).grid(row=0, column=6, padx=5)

        self.start_btn = ttk.Button(control_frame, text="Start Evolution", command=self.start_evolution)
        self.start_btn.grid(row=1, column=0, pady=(10, 0))

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_evolution)
        self.stop_btn.grid(row=1, column=1, padx=5, pady=(10, 0))

        self.test_btn = ttk.Button(control_frame, text="Test Historical Best", command=self.test_best_bot)
        self.test_btn.grid(row=1, column=2, padx=5, pady=(10, 0))

        self.save_chunks_btn = ttk.Button(control_frame, text="Save Chunk Bots", command=self.save_chunk_bots)
        self.save_chunks_btn.grid(row=1, column=3, padx=5, pady=(10, 0))

        self.test_best_chunk_btn = ttk.Button(control_frame, text="Test Best Chunk Bot", command=self.test_best_chunk_bot)
        self.test_best_chunk_btn.grid(row=1, column=4, padx=5, pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=1, column=5, columnspan=2, padx=(20, 0), sticky=tk.W)

        self.save_btn = ttk.Button(control_frame, text="Save Best Model", command=self.save_best_model)
        self.save_btn.grid(row=2, column=0, pady=(10, 0))

        self.load_btn = ttk.Button(control_frame, text="Load Saved Model", command=self.load_saved_model)
        self.load_btn.grid(row=2, column=1, padx=5, pady=(10, 0))

        self.vis_btn = ttk.Button(control_frame, text="Visualize Network", command=self.visualize_best_network)
        self.vis_btn.grid(row=2, column=2, padx=5, pady=(10, 0))

        # stats
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics & Trade Log", padding="10")
        stats_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.stats_text = tk.Text(stats_frame, height=20, width=50)
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.grid(row=0, column=0, sticky="nsew")
        stats_scroll.grid(row=0, column=1, sticky="ns")

        stats_frame.rowconfigure(0, weight=1)
        stats_frame.columnconfigure(0, weight=1)

        self.setup_plots(main_frame)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)


    def save_best_model(self):
        if self.historical_best_bot:
            self.historical_best_bot.save_model("./best_trading_model.pth")
            self.update_stats("✅ Saved best model to 'best_trading_model.pth'")
        else:
            self.update_stats("⚠️ No historical best bot to save.")

    def load_saved_model(self):
        bot = PyTorchTradingBot(
            input_shape=(self.ram_config.SEQUENCE_LENGTH, 17),  # Adjust shape as needed
            ram_config=self.ram_config
        )
        if bot.load_model("./best_trading_model.pth"):
            self.historical_best_bot = bot
            self.historical_best_fitness = 0.0  # Unknown fitness
            self.update_stats("✅ Loaded model as current best bot.")
        else:
            self.update_stats("⚠️ Failed to load saved model.")


    def setup_plots(self, parent):
        plot_frame = ttk.LabelFrame(parent, text="Performance Visualization", padding="10")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.clear_plots()

    def clear_plots(self):
        for ax in self.axes.flat:
            ax.clear()
        self.axes[0, 0].set_title('Evolution Progress')
        self.axes[0, 1].set_title('Portfolio Performance')
        self.axes[1, 0].set_title('Returns by Asset ($1000 start)')
        self.axes[1, 1].set_title('Memory Usage')
        self.canvas.draw()

    def update_stats(self, text):
        self.stats_text.insert(tk.END, text + "\n")
        self.stats_text.see(tk.END)
        self.root.update_idletasks()
    def update_stats_bot(self, text):
        self.stats_text.delete("end-2l", "end-1c")
        self.stats_text.insert(tk.END, text + "\n")
        self.stats_text.see(tk.END)
        self.root.update_idletasks()


    def update_plots(self):
        for ax in self.axes.flat:
            ax.clear()

        if len(self.evolution_history['best_fitness']) > 0:
            self.axes[0, 0].plot(self.evolution_history['generation'], self.evolution_history['best_fitness'], 'b-', label='Best Fitness', linewidth=2)
            self.axes[0, 0].plot(self.evolution_history['generation'], self.evolution_history['mean_fitness'], 'r--', label='Mean Fitness', alpha=0.7)
            self.axes[0, 0].set_title('Evolution Progress')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True)

            self.axes[0, 1].plot(self.evolution_history['generation'], self.evolution_history['best_return'], 'g-', linewidth=2)
            self.axes[0, 1].set_title('Best Return Evolution')
            self.axes[0, 1].grid(True)

        memory = psutil.virtual_memory()
        self.axes[1, 1].bar(['Used', 'Available'], [memory.percent, 100 - memory.percent], color=['red', 'green'])
        self.axes[1, 1].set_title(f'Memory Usage ({memory.percent:.1f}%)')
        self.axes[1, 1].set_ylim(0, 100)

        self.canvas.draw()

    def start_evolution(self):
        if self.evolution_running:
            return

        self.evolution_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.ram_config.POPULATION_SIZE = int(self.pop_var.get())
        self.ram_config.GENERATIONS = int(self.gen_var.get())
        self.ram_config.ELITE_SIZE = max(10, self.ram_config.POPULATION_SIZE // 4)

        tickers = get_random_tickers()

        self.evolution_thread = threading.Thread(target=self.run_evolution)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()

    def stop_evolution(self):
        self.evolution_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Stopped")

    def run_evolution(self):
        try:
            self.status_var.set("Loading data...")
            self.update_stats("Loading market data...")
            chunk_size = 13

            def load_dataset():
                try:
                    tickers = get_random_tickers(count=chunk_size, folder="./stock_data/train")
                    self.update_stats(f"training with: {tickers}")
                    return self.data_manager.load_essential_assets(
                        tickers=tickers,
                        use_csv=True,
                        csv_folder="./stock_data/train"
                    )
                except ValueError as e:
                    self.update_stats(f"{str(e)}")
                    return {}

            data_cache = load_dataset()
            if not data_cache:
                self.update_stats("error: No data loaded")
                self.stop_evolution()
                return

            self.update_stats(f"Loaded {len(data_cache)} assets: {list(data_cache.keys())}")
            first_asset = list(data_cache.keys())[0]
            sample_data = data_cache[first_asset]
            input_shape = (self.ram_config.SEQUENCE_LENGTH, sample_data.shape[1] + 4)

            population = [PyTorchTradingBot(input_shape, self.ram_config, layer_config=[128, 64, 32])
                        for _ in range(self.ram_config.POPULATION_SIZE)]
            self.update_stats(f"Created population of {len(population)} bots")

            self.evolution_history = {
                'best_fitness': [],
                'mean_fitness': [],
                'best_return': [],
                'generation': []
            }

            self.historical_best_bot = None
            self.historical_best_fitness = float('-inf')

            self.chunk_best_bots = []
            self.chunk_start_generation = 0
            self.chunk_best_fitness = float('-inf')
            self.chunk_best_bot = None

            for generation in range(self.ram_config.GENERATIONS):
                if generation % 5 == 0:
                    if self.chunk_best_bot:
                        self.chunk_best_bots.append(
                            (self.chunk_start_generation, generation - 1, self.chunk_best_bot)
                        )
                        self.update_stats(f"saved best bot from gen {self.chunk_start_generation} to {generation - 1}")

                    self.chunk_start_generation = generation
                    self.chunk_best_fitness = float('-inf')
                    self.chunk_best_bot = None

                    data_cache = load_dataset()
                    if not data_cache:
                        self.update_stats(f"failed to load new dataset at generation {generation}")
                        self.stop_evolution()
                        return
                    self.update_stats(f"generation {generation}: New training dataset: {list(data_cache.keys())}")

                if not self.evolution_running:
                    break

                self.status_var.set(f"generation {generation + 1}/{self.ram_config.GENERATIONS}")
                self.update_stats(f"generation {generation + 1}")

                results = []
                best_row_return = float('-inf')
                best_row_bot = None
                best_row_asset = None

                for i, bot in enumerate(population):
                    result = evaluate_bot_threaded(bot, data_cache, self.ram_config.EPISODES_PER_EVAL)
                    results.append(result)

                    _, _, _, _, asset, asset_ret = result
                    if asset_ret > best_row_return:
                        best_row_return = asset_ret
                        best_row_bot = result[0]
                        best_row_asset = asset

                    if i == 0:
                        self.update_stats(f"Evaluated {i + 1}/{len(population)} bots")
                    else:
                        self.update_stats_bot(f"Evaluated {i + 1}/{len(population)} bots")

                if not self.evolution_running:
                    break

                results.sort(key=lambda x: x[1], reverse=True)
                fitnesses = [r[1] for r in results]
                returns = [r[2] for r in results]

                best_fitness = fitnesses[0]
                mean_fitness = np.mean(fitnesses)
                best_return = returns[0]

                if best_fitness > self.chunk_best_fitness:
                    self.chunk_best_fitness = best_fitness
                    self.chunk_best_bot = results[0][0]

                if best_fitness > self.historical_best_fitness:
                    self.historical_best_fitness = best_fitness
                    self.historical_best_bot = results[0][0]
                    self.update_stats(f"new historical best fitness: {best_fitness:.4f}")

                self.evolution_history['best_fitness'].append(best_fitness)
                self.evolution_history['mean_fitness'].append(mean_fitness)
                self.evolution_history['best_return'].append(best_return)
                self.evolution_history['generation'].append(generation + 1)

                self.update_stats(f"best Fitness: {best_fitness:.4f}")
                self.update_stats(f"mean Fitness: {mean_fitness:.4f}")
                self.update_stats(f"best Return: {best_return:.2%}")
                self.update_stats(f"historical Best: {self.historical_best_fitness:.4f}")
                self.update_stats(f"best Bot from {best_row_asset}: Return {best_row_return:.2%}")

                self.update_plots()

                top5 = [r[0] for r in results[:5]]
                meta_tickers = list(data_cache.keys())[:3]  # use some for local adaptation

                mutated = [
                    best_row_bot.clone_and_mutate(
                        mutation_rate=0.02,
                        train_steps=2,
                        meta_tickers=meta_tickers
                    )
                    for _ in range(self.ram_config.POPULATION_SIZE - len(top5))
                ]

                population = top5 + mutated

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # save final chunk bot
            if self.chunk_best_bot:
                self.chunk_best_bots.append((self.chunk_start_generation, generation, self.chunk_best_bot))
                self.update_stats(f"saved final chunk bot: gen {self.chunk_start_generation} to {generation}")

            self.best_bots = [r[0] for r in results[:5]] if results else []
            self.update_stats(f"evolution completed")
            self.update_stats(f"historical best fitness: {self.historical_best_fitness:.4f}")
            self.update_stats(f"final generation best return: {max(returns) if returns else 0:.2%}")

        except Exception as e:
            self.update_stats(f"error: {str(e)}")
        finally:
            self.stop_evolution()


    def test_best_bot(self):
        self.status_var.set("downloading asset data...")
        self.update_stats("downloading asset data for testing...")

        chunk_size = 13  # define the number of assets to test on P.s. dont change coz you need to edit the whole fucking code

        try:
            tickers = get_random_tickers(count=chunk_size, folder="./stock_data/test")
            self.update_stats(f"🧪 Testing on: {tickers}")

            data_cache = self.data_manager.load_essential_assets(
                tickers=tickers, use_csv=True, csv_folder="./stock_data/test"
            )

            if not data_cache:
                self.update_stats("failed to load test data.")
                self.status_var.set("testing aborted")
                return

            self.update_stats(f"loaded {len(data_cache)} assets: {list(data_cache.keys())}")

            if not self.historical_best_bot:
                messagebox.showwarning("Warning", "⚠no trained bots available. Run evolution first.")
                return
            self.status_var.set("testing historical best bot...")

            best_bot = self.historical_best_bot
            test_results = {}
            total_days = 0

            self.update_stats(f"\n=== TESTING HISTORICAL BEST BOT (Fitness: {self.historical_best_fitness:.4f}) ===")
            self.update_stats(f"Starting capital: $1000 per asset")

            for ticker, data in data_cache.items():
                env = LightweightTradingEnv(
                    data, init_balance=1000, sequence_length=best_bot.ram_config.SEQUENCE_LENGTH
                )
                state = env.reset()
                done = False
                portfolio_values = [1000]
                actions_taken = []
                trading_days = len(data) - best_bot.ram_config.SEQUENCE_LENGTH

                while not done:
                    action = best_bot.act(state, deterministic=True)
                    state, _, done, info = env.step(action)
                    portfolio_values.append(info['total_value'])
                    actions_taken.append(action)

                final_value = portfolio_values[-1]
                final_return = (final_value - 1000) / 1000
                profit_loss = final_value - 1000

                test_results[ticker] = {
                    'return': final_return,
                    'values': portfolio_values,
                    'profit_loss': profit_loss,
                    'trades': info['total_trades'],
                    'trading_days': trading_days
                }

                total_days += trading_days

                self.update_stats(
                    f"{ticker:6}: {final_return:+7.2%} | ${profit_loss:+8.2f} | {info['total_trades']:3d} trades"
                )

            # Summary
            all_returns = [r['return'] for r in test_results.values()]
            avg_return = np.mean(all_returns)
            total_profit = sum(r['profit_loss'] for r in test_results.values())
            total_trades = sum(r['trades'] for r in test_results.values())

            self.update_stats(f"\n=== SUMMARY ===")
            self.update_stats(f"Average Return: {avg_return:+.2%}")
            self.update_stats(f"Total P&L: ${total_profit:+.2f}")
            self.update_stats(f"Total Trades: {total_trades}")
            self.update_stats(f"Assets Tested: {len(test_results)}")

            # Plot portfolio values
            self.axes[0, 1].clear()
            for ticker, result in test_results.items():
                self.axes[0, 1].plot(result['values'], label=f"{ticker} ({result['return']:+.1%})", alpha=0.7)
            self.axes[0, 1].set_title('Portfolio Performance ($1000 start)')
            self.axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.axes[0, 1].grid(True)

            # Bar chart: Returns by asset
            self.axes[1, 0].clear()
            final_tickers = list(test_results.keys())
            returns = [test_results[t]['return'] * 100 for t in final_tickers]
            colors = ['green' if r > 0 else 'red' for r in returns]
            self.axes[1, 0].bar(final_tickers, returns, color=colors, alpha=0.7)
            self.axes[1, 0].set_title('Returns by Asset (%)')
            self.axes[1, 0].tick_params(axis='x', rotation=45)
            self.axes[1, 0].grid(True, alpha=0.3)

            self.canvas.draw()
            self.status_var.set("testing completed")

        except Exception as e:
            self.update_stats(f"error during testing: {str(e)}")
            self.status_var.set("Testing failed")

    def test_best_chunk_bot(self):
        if not self.chunk_best_bots:
            messagebox.showwarning("No Bots", "No chunk-best bots available to test.")
            return

        self.status_var.set("testing best chunk bot...")
        self.update_stats("downloading asset data for testing (best chunk bot)...")

        chunk_size = 13

        try:
            tickers = get_random_tickers(count=chunk_size, folder="./stock_data/test")
            self.update_stats(f"testing on: {tickers}")

            data_cache = self.data_manager.load_essential_assets(
                tickers=tickers, use_csv=True, csv_folder="./stock_data/test"
            )

            if not data_cache:
                self.update_stats("failed to load test data.")
                self.status_var.set("Testing aborted")
                return

            self.update_stats(f"loaded {len(data_cache)} assets: {list(data_cache.keys())}")

            # Select best chunk bot by evaluating historical fitness
            best_entry = max(self.chunk_best_bots, key=lambda x: evaluate_bot_threaded(x[2], data_cache, 1)[1])
            start, end, best_bot = best_entry

            self.status_var.set(f"Testing best chunk bot from Gen {start}–{end}...")
            self.update_stats(f"\n=== TESTING BEST CHUNK BOT (Gen {start}–{end}) ===")
            self.update_stats(f"Starting capital: $1000 per asset")

            test_results = {}
            for ticker, data in data_cache.items():
                env = LightweightTradingEnv(data, init_balance=1000, sequence_length=best_bot.ram_config.SEQUENCE_LENGTH)
                state = env.reset()
                done = False
                portfolio_values = [1000]
                actions_taken = []

                while not done:
                    action = best_bot.act(state, deterministic=True)
                    state, _, done, info = env.step(action)
                    portfolio_values.append(info['total_value'])
                    actions_taken.append(action)

                final_value = portfolio_values[-1]
                final_return = (final_value - 1000) / 1000
                profit_loss = final_value - 1000

                test_results[ticker] = {
                    'return': final_return,
                    'values': portfolio_values,
                    'profit_loss': profit_loss,
                    'trades': info['total_trades'],
                    'trading_days': len(data) - best_bot.ram_config.SEQUENCE_LENGTH
                }

                self.update_stats(
                    f"{ticker:6}: {final_return:+7.2%} | ${profit_loss:+8.2f} | {info['total_trades']:3d} trades"
                )

            # Summary
            all_returns = [r['return'] for r in test_results.values()]
            avg_return = np.mean(all_returns)
            total_profit = sum(r['profit_loss'] for r in test_results.values())
            total_trades = sum(r['trades'] for r in test_results.values())

            self.update_stats(f"\n=== SUMMARY ===")
            self.update_stats(f"Average Return: {avg_return:+.2%}")
            self.update_stats(f"Total P&L: ${total_profit:+.2f}")
            self.update_stats(f"Total Trades: {total_trades}")
            self.update_stats(f"Assets Tested: {len(test_results)}")

            # Plot portfolio values
            self.axes[0, 1].clear()
            for ticker, result in test_results.items():
                self.axes[0, 1].plot(result['values'], label=f"{ticker} ({result['return']:+.1%})", alpha=0.7)
            self.axes[0, 1].set_title('Best Chunk Bot: Portfolio Performance ($1000 start)')
            self.axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.axes[0, 1].grid(True)

            # Bar chart: Returns by asset
            self.axes[1, 0].clear()
            final_tickers = list(test_results.keys())
            returns = [test_results[t]['return'] * 100 for t in final_tickers]
            colors = ['green' if r > 0 else 'red' for r in returns]
            self.axes[1, 0].bar(final_tickers, returns, color=colors, alpha=0.7)
            self.axes[1, 0].set_title('Best Chunk Bot: Returns by Asset (%)')
            self.axes[1, 0].tick_params(axis='x', rotation=45)
            self.axes[1, 0].grid(True, alpha=0.3)

            self.canvas.draw()
            self.status_var.set("testing completed")

        except Exception as e:
            self.update_stats(f"error during best chunk bot test: {str(e)}")
            self.status_var.set("testing failed")



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TradingBotUI()
    app.run()
