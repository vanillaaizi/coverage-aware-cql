"""
IE540 Dynamic Programming & Reinforcement Learning term project toy experiment.
Topic: Coverage-Aware Conservative Q-Learning for Offline RL.

How to run:
    python run_ie540_cql_gridworld.py

Outputs:
    results_summary.csv
    fig_success_rate.png
    fig_average_return.png
    fig_paths.png
"""

import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)


class GridWorld:
    """Small deterministic gridworld for illustrating offline RL distribution shift."""

    def __init__(self, n=7):
        self.n = n
        self.start = (n - 1, 0)
        self.goal = (0, n - 1)
        self.obstacles = {(3, 1), (3, 2), (3, 3), (1, 3), (2, 3), (4, 5)}
        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ["U", "D", "L", "R"]
        self.A = 4

    @property
    def S(self):
        return self.n * self.n

    def state_to_idx(self, s):
        return s[0] * self.n + s[1]

    def idx_to_state(self, i):
        return (i // self.n, i % self.n)

    def step(self, s, a):
        if s == self.goal:
            return s, 0.0, True
        dr, dc = self.actions[a]
        ns = (s[0] + dr, s[1] + dc)
        if not (0 <= ns[0] < self.n and 0 <= ns[1] < self.n) or ns in self.obstacles:
            # hitting wall/obstacle is worse
            return s, -2.0, False
        done = ns == self.goal
        # Use non-positive rewards so unseen zero-valued actions look falsely attractive.
        reward = 0.0 if done else -1.0
        return ns, reward, done


def expert_action(env, s):
    scores = []
    for a in range(env.A):
        ns, _, _ = env.step(s, a)
        manhattan = abs(ns[0] - env.goal[0]) + abs(ns[1] - env.goal[1])
        blocked_penalty = 5 if ns == s else 0
        scores.append(-manhattan - blocked_penalty)
    return int(np.argmax(scores))


def behavior_action(env, s, quality=0.95):
    """Dataset policy: mostly follows a good path, sometimes explores."""
    if random.random() < quality:
        return expert_action(env, s)
    # Mild exploration, biased toward up/right to keep dataset plausible.
    return random.choice([0, 3, random.randrange(env.A)])


def collect_dataset(env, episodes=50, quality=0.95, max_steps=30, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    data, returns, successes = [], [], []
    for _ in range(episodes):
        s = env.start
        G = 0.0
        done = False
        for _ in range(max_steps):
            a = behavior_action(env, s, quality)
            ns, r, done = env.step(s, a)
            data.append((env.state_to_idx(s), a, r, env.state_to_idx(ns), done))
            G += r
            s = ns
            if done:
                break
        returns.append(G)
        successes.append(done)
    return data, returns, successes


def train_offline_q(data, env, method="offline_q", epochs=80, lr=0.05, gamma=0.97, alpha=0.50, seed=0):
    """
    Three variants:
    - offline_q: standard offline Q-learning using only fixed data.
    - cql: Bellman update + fixed CQL-style logsumexp penalty.
    - cov_cql: Bellman update + stronger CQL penalty for low-coverage states.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.S, env.A))
    state_counts = Counter(s for s, _, _, _, _ in data)
    mean_count = np.mean(list(state_counts.values()))

    for _ in range(epochs):
        for i in rng.permutation(len(data)):
            s, a, r, ns, done = data[i]

            # Standard Bellman optimality backup.
            target = r + (0.0 if done else gamma * np.max(Q[ns]))
            Q[s, a] += lr * (target - Q[s, a])

            if method in {"cql", "cov_cql"}:
                if method == "cql":
                    local_alpha = alpha
                else:
                    # Own extension: adapt conservatism to dataset coverage.
                    # Rare states get stronger pessimism; frequent states get weaker pessimism.
                    local_alpha = alpha * math.sqrt(mean_count / (state_counts[s] + 1))
                    local_alpha = float(np.clip(local_alpha, 0.05, 1.20))

                # Gradient of alpha * (logsumexp_a Q(s,a) - Q(s,a_data)).
                # This suppresses unsupported high-valued actions and raises dataset actions.
                x = Q[s] - np.max(Q[s])
                softmax = np.exp(x) / np.sum(np.exp(x))
                Q[s, :] -= lr * local_alpha * softmax
                Q[s, a] += lr * local_alpha
    return Q


def evaluate_policy(Q, env, episodes=100, max_steps=40):
    returns, successes, lengths = [], [], []
    example_path = None
    for _ in range(episodes):
        s = env.start
        G = 0.0
        path = [s]
        done = False
        for t in range(max_steps):
            a = int(np.argmax(Q[env.state_to_idx(s)]))
            ns, r, done = env.step(s, a)
            G += r
            s = ns
            path.append(s)
            if done:
                lengths.append(t + 1)
                break
        if not done:
            lengths.append(max_steps)
        returns.append(G)
        successes.append(done)
        if example_path is None:
            example_path = path
    return float(np.mean(returns)), float(np.mean(successes)), float(np.mean(lengths)), example_path


def run_experiment(seeds=range(8)):
    rows = []
    all_paths = {}
    method_names = {
        "offline_q": "Offline Q-learning",
        "cql": "Fixed-alpha CQL",
        "cov_cql": "Coverage-aware CQL",
    }

    for seed in seeds:
        env = GridWorld()
        data, behavior_returns, behavior_successes = collect_dataset(env, episodes=50, quality=0.95, seed=seed)
        coverage = len(set((s, a) for s, a, _, _, _ in data)) / (env.S * env.A)

        for method in ["offline_q", "cql", "cov_cql"]:
            Q = train_offline_q(data, env, method=method, seed=seed)
            avg_return, success_rate, avg_len, path = evaluate_policy(Q, env)
            rows.append(
                {
                    "seed": seed,
                    "method": method_names[method],
                    "average_return": avg_return,
                    "success_rate": success_rate,
                    "average_episode_length": avg_len,
                    "dataset_state_action_coverage": coverage,
                    "behavior_policy_success_rate": np.mean(behavior_successes),
                    "behavior_policy_average_return": np.mean(behavior_returns),
                }
            )
            if seed == 0:
                all_paths[method_names[method]] = path

    return pd.DataFrame(rows), all_paths


def save_barplot(summary, metric, ylabel, filename):
    plt.figure(figsize=(7.2, 4.4))
    order = ["Offline Q-learning", "Fixed-alpha CQL", "Coverage-aware CQL"]
    labels = [m for m in order if m in summary.index]
    values = summary.loc[labels, (metric, "mean")].values
    x = np.arange(len(labels))
    bars = plt.bar(x, values)
    plt.xticks(x, labels, rotation=8, ha="center")
    plt.ylabel(ylabel)
    plt.title(ylabel + " across 8 random dataset seeds")
    if metric == "success_rate":
        plt.ylim(0, 1.05)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=220)
    plt.close()


def save_paths(paths, filename="fig_paths.png"):
    env = GridWorld()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, path) in zip(axes, paths.items()):
        ax.set_xlim(-0.5, env.n - 0.5)
        ax.set_ylim(env.n - 0.5, -0.5)
        ax.set_xticks(range(env.n))
        ax.set_yticks(range(env.n))
        ax.grid(True)
        for r, c in env.obstacles:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.35))
        ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontsize=12, fontweight="bold")
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=220)
    plt.close()


if __name__ == "__main__":
    df, paths = run_experiment()
    df.to_csv(OUT / "results_raw.csv", index=False)
    summary = df.groupby("method").agg({
        "average_return": ["mean", "std"],
        "success_rate": ["mean", "std"],
        "average_episode_length": ["mean", "std"],
        "dataset_state_action_coverage": ["mean", "std"],
        "behavior_policy_success_rate": ["mean", "std"],
    })
    summary.to_csv(OUT / "results_summary.csv")
    save_barplot(summary, "success_rate", "Success rate", "fig_success_rate.png")
    save_barplot(summary, "average_return", "Average return", "fig_average_return.png")
    save_paths(paths)
    print(summary.round(3))
    print("\nFiles saved to:", OUT)
