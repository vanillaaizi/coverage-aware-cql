import numpy as np

from src.experiment import GridWorld, collect_dataset, train_offline_q


def test_gridworld_terminal_reward():
    env = GridWorld()
    s = (0, env.n - 2)
    ns, reward, done = env.step(s, 3)
    assert ns == env.goal
    assert reward == 0.0
    assert done is True


def test_dataset_generation_is_reproducible():
    env = GridWorld()
    d1, r1, s1 = collect_dataset(env, episodes=5, seed=7)
    d2, r2, s2 = collect_dataset(env, episodes=5, seed=7)
    assert d1 == d2
    assert r1 == r2
    assert s1 == s2


def test_all_three_training_modes_return_valid_q_tables():
    env = GridWorld()
    data, _, _ = collect_dataset(env, episodes=5, seed=0)
    for method in ("offline_q", "cql", "cov_cql"):
        q = train_offline_q(data, env, method=method, epochs=2, seed=0)
        assert q.shape == (env.S, env.A)
        assert np.isfinite(q).all()
