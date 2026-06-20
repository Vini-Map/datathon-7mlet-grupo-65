"""Neural contextual bandit in PyTorch (Stage 3).

A small MLP estimates P(reward | context) per arm; exploration is epsilon-greedy
with a decaying schedule. Observations are stored in a replay buffer and the
network is refit periodically on mini-batches (BCE on the taken arm only). This
is the project's PyTorch component — it shows the context entering the decision
through a learned non-linear reward model, complementing the linear LinUCB.

Cold-start: while the buffer is small, epsilon is near 1 (mostly exploring) and
predictions are uninformative, so the policy behaves like random until it has
seen enough feedback.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from aep.bandits.base import Context, Policy


class _RewardNet(nn.Module):
    def __init__(self, dim: int, n_arms: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_arms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits per arm
        return self.net(x)


class NeuralBandit(Policy):
    name = "neural"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        rng: np.random.Generator | None = None,
        seed: int = 0,
        hidden: int = 32,
        lr: float = 0.01,
        train_every: int = 64,
        batch_size: int = 128,
        grad_steps: int = 2,
        eps0: float = 1.0,
        eps_min: float = 0.03,
        eps_decay: float = 0.9995,
    ) -> None:
        super().__init__(n_arms, rng)
        self.dim = dim
        self.train_every = train_every
        self.batch_size = batch_size
        self.grad_steps = grad_steps
        self.eps0 = eps0
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self._lr = lr
        self._hidden = hidden
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        torch.manual_seed(self._seed)
        self.net = _RewardNet(self.dim, self.n_arms, self._hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self._lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._buf_x: list[np.ndarray] = []
        self._buf_a: list[int] = []
        self._buf_r: list[float] = []
        self._steps = 0
        self.epsilon = self.eps0

    def select(self, ctx: Context) -> int:
        self.epsilon = max(self.eps_min, self.eps0 * self.eps_decay**self._steps)
        if self.rng.random() < self.epsilon or len(self._buf_r) < self.batch_size:
            return int(self.rng.choice(ctx.eligible_idx))
        with torch.no_grad():
            x = torch.tensor(ctx.x, dtype=torch.float32).unsqueeze(0)
            scores = torch.sigmoid(self.net(x)).squeeze(0).numpy()
        return self._argmax_eligible(scores, ctx.eligible)

    def update(self, arm: int, reward: float, ctx: Context) -> None:
        self._buf_x.append(ctx.x.astype(np.float32))
        self._buf_a.append(arm)
        self._buf_r.append(float(reward))
        self._steps += 1
        if self._steps % self.train_every == 0 and len(self._buf_r) >= self.batch_size:
            self._train()

    def _train(self) -> None:
        X = torch.tensor(np.array(self._buf_x), dtype=torch.float32)
        a = torch.tensor(self._buf_a, dtype=torch.long)
        r = torch.tensor(self._buf_r, dtype=torch.float32)
        n = len(self._buf_r)
        for _ in range(self.grad_steps):
            idx = torch.from_numpy(self.rng.choice(n, size=self.batch_size, replace=False))
            logits = self.net(X[idx])
            pred = logits.gather(1, a[idx].unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(pred, r[idx])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
