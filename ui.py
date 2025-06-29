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
        self.POPULATION_SIZE = 30
        self.ELITE_SIZE = 10
        self.MEMORY_PER_BOT = 2000
        self.BATCH_SIZE = 32
        self.SEQUENCE_LENGTH = 20
        self.PARALLEL_WORKERS = 2
        self.GENERATIONS = 15
        self.EPISODES_PER_EVAL = 3
        self.MAX_CACHED_ASSETS = 3
        self.DATA_CHUNK_SIZE = 500
        self.LEARNING_RATE = 0.001
        self.GAMMA = 0.95

class LightweightDataManager:
    def __init__(self, ram_config):
        self.ram_config = ram_config
        self.data_cache = {}
        self.current_cached_assets = 0

    def load_essential_assets(self, tickers=['SPY', 'QQQ', 'AAPL'], start="2020-01-01", end="2024-01-01"):
        self.data_cache = {}
        self.current_cached_assets = 0

        for ticker in tickers[:self.ram_config.MAX_CACHED_ASSETS]:
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
    def __init__(self, df, init_balance=10000, transaction_cost=0.001, sequence_length=20):
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
        self.reset()

    def reset(self):
        self.current_step = self.sequence_length
        self.balance = self.init_balance
        self.shares_held = 0
        self.total_value = self.init_balance
        self.max_net_worth = self.init_balance
        self.total_trades = 0
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
            (self.total_value - self.init_balance) / self.init_balance
        ], dtype=np.float32)
        return np.concatenate([market_data, portfolio_data])

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return np.array(list(self.sequence_buffer)), 0, True, self._get_info()
        current_price = float(self.df.iloc[self.current_step]['Close'])
        prev_value = self.total_value
        if action == 0:
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
                self.total_trades += 1
        elif action == 2:
            if self.balance > 0:
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.shares_held += shares_to_buy
                self.balance -= cost
                self.total_trades += 1
        self.current_step += 1
        obs = self._get_observation(self.current_step)
        self.sequence_buffer.append(obs)
        reward = self._calculate_reward(prev_value)
        self.max_net_worth = max(self.max_net_worth, self.total_value)
        done = self.current_step >= len(self.df) - 1
        return np.array(list(self.sequence_buffer)), reward, done, self._get_info()

    def _calculate_reward(self, prev_value):
        if prev_value <= 0:
            return 0
        value_change = self.total_value - prev_value
        portfolio_return = value_change / prev_value
        if portfolio_return > 0:
            reward = portfolio_return * 150
        else:
            reward = portfolio_return * 200
        drawdown = (self.max_net_worth - self.total_value) / self.max_net_worth if self.max_net_worth > 0 else 0
        reward -= drawdown * 10
        reward += -0.005 if self.total_trades > 0 else 0
        return reward

    def _get_info(self):
        return {
            'total_value': self.total_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'portfolio_return': (self.total_value - self.init_balance) / self.init_balance
        }

class TradingNetwork(nn.Module):
    def __init__(self, input_shape, hidden_size=32):
        super(TradingNetwork, self).__init__()
        sequence_length, feature_size = input_shape
        self.lstm1 = nn.LSTM(feature_size, hidden_size, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_size//2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class PyTorchTradingBot:
    def __init__(self, input_shape, ram_config, learning_rate=0.001):
        self.input_shape = input_shape
        self.ram_config = ram_config
        self.device = device
        self.network = TradingNetwork(input_shape).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = deque(maxlen=ram_config.MEMORY_PER_BOT)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.fitness_history = []
        self.generation = 0

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

    def clone_and_mutate(self, mutation_rate=0.1):
        child = PyTorchTradingBot(self.input_shape, self.ram_config)
        with torch.no_grad():
            for child_param, parent_param in zip(child.network.parameters(), self.network.parameters()):
                if np.random.random() < 0.7:
                    mutation_strength = np.random.choice([0.05, 0.1, 0.2], p=[0.3, 0.5, 0.2])
                    noise = torch.randn_like(parent_param) * mutation_strength
                    child_param.data = parent_param.data + noise
                else:
                    child_param.data = parent_param.data.clone()
        child.epsilon = self.epsilon * np.random.uniform(0.9, 1.1)
        child.generation = self.generation + 1
        return child

def evaluate_bot_threaded(bot, data_dict, episodes=2):
    results = []
    assets = list(data_dict.keys())
    selected_assets = random.sample(assets, min(2, len(assets)))
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
        if len(bot.memory) >= bot.ram_config.BATCH_SIZE:
            bot.train_batch()
        final_return = info['portfolio_return']
        results.append(final_return)
    mean_return = np.mean(results)
    std_return = np.std(results) if len(results) > 1 else 0
    fitness = mean_return - 0.1 * std_return
    return bot, fitness, mean_return, results

class TradingBotUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PyTorch Trading Bot Visualizer")
        self.root.geometry("1400x900")

        self.ram_config = OptimizedRAMConfig()
        self.data_manager = LightweightDataManager(self.ram_config)
        self.evolution_running = False
        self.evolution_thread = None
        self.best_bots = []

        self.evolution_history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_return': [],
            'generation': []
        }

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(control_frame, text="Tickers:").grid(row=0, column=0, sticky=tk.W)
        self.ticker_entry = ttk.Entry(control_frame, width=20)
        self.ticker_entry.insert(0, "SPY,QQQ,AAPL,BTC")
        self.ticker_entry.grid(row=0, column=1, padx=(10, 0))

        ttk.Label(control_frame, text="Population:").grid(row=0, column=2, padx=(20, 0), sticky=tk.W)
        self.pop_var = tk.StringVar(value="30")
        ttk.Entry(control_frame, textvariable=self.pop_var, width=10).grid(row=0, column=3, padx=(10, 0))

        ttk.Label(control_frame, text="Generations:").grid(row=0, column=4, padx=(20, 0), sticky=tk.W)
        self.gen_var = tk.StringVar(value="15")
        ttk.Entry(control_frame, textvariable=self.gen_var, width=10).grid(row=0, column=5, padx=(10, 0))

        self.start_btn = ttk.Button(control_frame, text="Start Evolution", command=self.start_evolution)
        self.start_btn.grid(row=1, column=0, pady=(10, 0))

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_evolution)
        self.stop_btn.grid(row=1, column=1, padx=(10, 0), pady=(10, 0))

        self.test_btn = ttk.Button(control_frame, text="Test Best Bot", command=self.test_best_bot)
        self.test_btn.grid(row=1, column=2, padx=(10, 0), pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=1, column=3, columnspan=3, padx=(20, 0), pady=(10, 0), sticky=tk.W)

        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=15, width=40)
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.grid(row=0, column=0)
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.setup_plots(main_frame)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def setup_plots(self, parent):
        plot_frame = ttk.LabelFrame(parent, text="Performance Visualization", padding="10")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.clear_plots()

    def clear_plots(self):
        for ax in self.axes.flat:
            ax.clear()
        self.axes[0, 0].set_title('Evolution Progress')
        self.axes[0, 1].set_title('Portfolio Performance')
        self.axes[1, 0].set_title('Returns by Asset')
        self.axes[1, 1].set_title('Memory Usage')
        self.canvas.draw()

    def update_stats(self, text):
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
        self.ram_config.ELITE_SIZE = max(5, self.ram_config.POPULATION_SIZE // 3)

        tickers = [t.strip() for t in self.ticker_entry.get().split(',')]

        self.evolution_thread = threading.Thread(target=self.run_evolution, args=(tickers,))
        self.evolution_thread.daemon = True
        self.evolution_thread.start()

    def stop_evolution(self):
        self.evolution_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Stopped")

    def run_evolution(self, tickers):
        try:
            self.status_var.set("Loading data...")
            self.update_stats("Loading market data...")

            data_cache = self.data_manager.load_essential_assets(tickers)
            if not data_cache:
                self.update_stats("Error: No data loaded")
                self.stop_evolution()
                return

            self.update_stats(f"Loaded {len(data_cache)} assets")

            first_asset = list(data_cache.keys())[0]
            sample_data = data_cache[first_asset]
            input_shape = (self.ram_config.SEQUENCE_LENGTH, sample_data.shape[1] + 4)

            population = []
            for i in range(self.ram_config.POPULATION_SIZE):
                bot = PyTorchTradingBot(input_shape, self.ram_config)
                population.append(bot)

            self.update_stats(f"Created population of {len(population)} bots")

            self.evolution_history = {
                'best_fitness': [],
                'mean_fitness': [],
                'best_return': [],
                'generation': []
            }

            for generation in range(self.ram_config.GENERATIONS):
                if not self.evolution_running:
                    break

                self.status_var.set(f"Generation {generation + 1}/{self.ram_config.GENERATIONS}")
                self.update_stats(f"\nGeneration {generation + 1}")

                results = []
                for i, bot in enumerate(population):
                    if not self.evolution_running:
                        break
                    result = evaluate_bot_threaded(bot, data_cache, self.ram_config.EPISODES_PER_EVAL)
                    results.append(result)

                    if (i + 1) % 5 == 0:
                        self.update_stats(f"  Evaluated {i + 1}/{len(population)} bots")

                if not self.evolution_running:
                    break

                results.sort(key=lambda x: x[1], reverse=True)

                fitnesses = [r[1] for r in results]
                returns = [r[2] for r in results]

                best_fitness = fitnesses[0]
                mean_fitness = np.mean(fitnesses)
                best_return = returns[0]

                self.evolution_history['best_fitness'].append(best_fitness)
                self.evolution_history['mean_fitness'].append(mean_fitness)
                self.evolution_history['best_return'].append(best_return)
                self.evolution_history['generation'].append(generation + 1)

                self.update_stats(f"  Best Fitness: {best_fitness:.4f}")
                self.update_stats(f"  Mean Fitness: {mean_fitness:.4f}")
                self.update_stats(f"  Best Return: {best_return:.2%}")

                self.update_plots()

                elite_bots = [r[0] for r in results[:self.ram_config.ELITE_SIZE]]
                new_population = [bot.clone_and_mutate(mutation_rate=0.01) for bot in elite_bots]

                while len(new_population) < self.ram_config.POPULATION_SIZE:
                    tournament_size = min(4, len(elite_bots))
                    tournament = random.sample(elite_bots, tournament_size)
                    parent = random.choice(tournament)
                    child = parent.clone_and_mutate()
                    new_population.append(child)

                population = new_population

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            self.best_bots = [r[0] for r in results[:5]] if results else []
            self.update_stats(f"\nEvolution completed!")
            self.update_stats(f"Best overall return: {max(returns) if returns else 0:.2%}")

        except Exception as e:
            self.update_stats(f"Error: {str(e)}")
        finally:
            self.stop_evolution()

    def test_best_bot(self):
        if not self.best_bots:
            messagebox.showwarning("Warning", "No trained bots available. Run evolution first.")
            return

        self.status_var.set("Testing best bot...")

        try:
            best_bot = self.best_bots[0]
            test_results = {}

            for ticker, data in self.data_manager.data_cache.items():
                env = LightweightTradingEnv(data, sequence_length=best_bot.ram_config.SEQUENCE_LENGTH)
                state = env.reset()
                done = False
                portfolio_values = [env.init_balance]
                actions_taken = []

                while not done:
                    action = best_bot.act(state, deterministic=True)
                    state, _, done, info = env.step(action)
                    portfolio_values.append(info['total_value'])
                    actions_taken.append(action)

                final_return = (portfolio_values[-1] - env.init_balance) / env.init_balance
                test_results[ticker] = {
                    'return': final_return,
                    'values': portfolio_values,
                    'actions': actions_taken
                }

            self.axes[1, 0].clear()
            assets = list(test_results.keys())
            returns = [test_results[a]['return'] for a in assets]
            colors = ['green' if r > 0 else 'red' for r in returns]
            bars = self.axes[1, 0].bar(assets, returns, color=colors, alpha=0.7)
            self.axes[1, 0].set_title('Test Results - Returns by Asset')
            self.axes[1, 0].set_ylabel('Return (%)')

            for bar, ret in zip(bars, returns):
                height = bar.get_height()
                self.axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.1%}', ha='center', va='bottom' if height > 0 else 'top')

            self.axes[0, 1].clear()
            for ticker, result in test_results.items():
                self.axes[0, 1].plot(result['values'], label=f"{ticker} ({result['return']:.1%})", alpha=0.8)
            self.axes[0, 1].set_title('Portfolio Performance - Test Run')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True)

            self.canvas.draw()

            self.update_stats(f"\nTest Results:")
            for ticker, result in test_results.items():
                self.update_stats(f"  {ticker}: {result['return']:.2%}")

            self.status_var.set("Test completed")

        except Exception as e:
            self.update_stats(f"Test error: {str(e)}")
            self.status_var.set("Test failed")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TradingBotUI()
    app.run()
