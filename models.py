# -*- coding: utf-8 -*-
"""
세 가지 모델 계열을 한 번에 학습하는 모듈

  1) 선형회귀        y = b0 + Σ bi·xi                 (최소제곱법)
  2) 비선형회귀
       2차 다항      y = c0 + Σ (ci·xi + di·xi²)       (계수에 대해선 선형)
       거듭제곱      y = a · Π xi^bi                   (scipy curve_fit)
       지수          y = a · e^(Σ bi·xi)               (scipy curve_fit)
  3) 인공신경망      입력 → 32(ReLU) → 16(ReLU) → 1    (NumPy 순전파·역전파·Adam)

기존 저장소의 regression.py(키·몸무게)와 ann_diabetes_predictor.py(당뇨병)의 식과
학습 방법을 그대로 가져와, 입력 열이 몇 개든 동작하도록 일반화했습니다.
fit_all() 한 번 호출로 모든 모델이 같은 학습 데이터로 학습되고 같은 테스트 데이터로 평가됩니다.
"""
import warnings

import numpy as np
from scipy.optimize import curve_fit

SEED = 42
HIDDEN = [32, 16]
EPOCHS, BATCH, LR, L2, PATIENCE = 1000, 32, 1e-3, 1e-3, 50


# ── 평가 지표 ─────────────────────────────────────────
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2_score(y, yhat):
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def split_indices(n, seed=SEED):
    """학습 70% / 검증 15% / 테스트 15%.
    기존 ann_diabetes_predictor.py의 train_test_split(random_state=42) 두 번과 같은 행이 뽑히도록
    scikit-learn의 섞는 순서를 NumPy로 그대로 따른다 (scikit-learn 설치 불필요)."""
    n_tmp = int(np.ceil(n * 0.30))
    perm = np.random.RandomState(seed).permutation(n)
    tmp, tr = perm[:n_tmp], perm[n_tmp:]
    n_te = int(np.ceil(n_tmp * 0.50))
    perm2 = np.random.RandomState(seed).permutation(n_tmp)
    return tr, tmp[perm2[n_te:]], tmp[perm2[:n_te]]


def _fmt(v):
    return f"{v:.4g}"


def _terms(coefs, names, suffix=""):
    return " ".join(f"{'+' if c >= 0 else '−'} {_fmt(abs(c))}·{n}{suffix}" for c, n in zip(coefs, names))


# ── 1. 선형회귀 ───────────────────────────────────────
class LinearModel:
    name = "선형회귀"
    family = "선형"

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.b0, self.b = float(beta[0]), beta[1:]
        return self

    def predict(self, X):
        return self.b0 + X @ self.b

    def equation(self, names):
        return f"y = {_fmt(self.b0)} {_terms(self.b, names)}"


# ── 2-1. 2차 다항회귀 ─────────────────────────────────
class Poly2Model:
    name = "비선형(2차 다항)"
    family = "비선형"

    def fit(self, X, y):
        # 표준화한 값으로 풀고(수치 안정), 계수는 원래 단위로 되돌려 저장
        m, s = X.mean(0), X.std(0)
        Z = (X - m) / s
        # 값이 두 종류뿐인 열(예: 성별 1, 2)은 x²이 x와 겹치므로 제곱항을 넣지 않는다
        sq = np.array([len(np.unique(col)) > 2 for col in X.T])
        A = np.column_stack([np.ones(len(Z)), Z, Z[:, sq] ** 2])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        k = X.shape[1]
        c1, c2 = beta[1:1 + k], np.zeros(k)
        c2[sq] = beta[1 + k:]
        self.c2 = c2 / s ** 2
        self.c1 = c1 / s - 2 * c2 * m / s ** 2
        self.c0 = float(beta[0] - np.sum(c1 * m / s) + np.sum(c2 * m ** 2 / s ** 2))
        return self

    def predict(self, X):
        return self.c0 + X @ self.c1 + (X ** 2) @ self.c2

    def equation(self, names):
        sq = [(c, n) for c, n in zip(self.c2, names) if c != 0]
        return (f"y = {_fmt(self.c0)} {_terms(self.c1, names)} "
                f"{_terms([c for c, _ in sq], [n for _, n in sq], '²')}")


# ── 2-2. 비선형회귀: 거듭제곱 ─────────────────────────
class PowerModel:
    name = "비선형(거듭제곱)"
    family = "비선형"

    @staticmethod
    def _f(X, *p):
        return p[0] * np.prod(np.power(X, np.asarray(p[1:])), axis=1)

    def fit(self, X, y):
        if np.any(X <= 0) or np.any(y <= 0):
            raise ValueError("0 이하의 값이 있어 거듭제곱 모델을 쓸 수 없습니다")
        # 초기값: log y = log a + Σ b·log x 직선의 계수
        A = np.column_stack([np.ones(len(X)), np.log(X)])
        beta, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        p0 = [np.exp(beta[0]), *beta[1:]]
        p, _ = curve_fit(self._f, X, y, p0=p0, maxfev=20000)
        self.a, self.b = float(p[0]), np.asarray(p[1:])
        return self

    def predict(self, X):
        Xc = np.where(X > 0, X, np.nan)      # 0 이하 입력은 계산 불가
        return self._f(Xc, self.a, *self.b)

    def equation(self, names):
        return f"y = {_fmt(self.a)} · " + " · ".join(f"{n}^{_fmt(b)}" for b, n in zip(self.b, names))


# ── 2-3. 비선형회귀: 지수 ─────────────────────────────
class ExpModel:
    name = "비선형(지수)"
    family = "비선형"

    @staticmethod
    def _f(X, *p):
        return p[0] * np.exp(X @ np.asarray(p[1:]))

    def fit(self, X, y):
        if np.any(y <= 0):
            raise ValueError("목표값에 0 이하의 값이 있어 지수 모델을 쓸 수 없습니다")
        # 초기값: log y = log a + Σ b·x 직선의 계수
        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        p0 = [np.exp(beta[0]), *beta[1:]]
        p, _ = curve_fit(self._f, X, y, p0=p0, maxfev=20000)
        self.a, self.b = float(p[0]), np.asarray(p[1:])
        return self

    def predict(self, X):
        return self._f(X, self.a, *self.b)

    def equation(self, names):
        inner = _terms(self.b, names).lstrip("+ ").strip()
        return f"y = {_fmt(self.a)} · e^({inner})"


# ── 3. 인공신경망 (MLP) ───────────────────────────────
class MLPModel:
    name = "인공신경망"
    family = "인공신경망"

    def __init__(self, hidden=HIDDEN, l2=L2, seed=SEED):
        self.hidden, self.l2 = hidden, l2
        self.rng = np.random.default_rng(seed)

    def _init(self, n_in):
        sizes = [n_in, *self.hidden, 1]
        self.sizes = sizes
        self.W, self.B = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            # He 초기화: ReLU 층에서 신호 분산이 유지되도록 표준편차 sqrt(2/n_in)
            self.W.append(self.rng.normal(0, np.sqrt(2.0 / a), (a, b)))
            self.B.append(np.zeros((1, b)))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mB = [np.zeros_like(b) for b in self.B]
        self.vB = [np.zeros_like(b) for b in self.B]
        self.t = 0

    def _forward(self, X):
        """순전파: z = aW + b, 은닉층은 ReLU, 출력층은 선형"""
        self.cache = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.B)):
            z = a @ W + b
            a = z if i == len(self.W) - 1 else np.maximum(0, z)
            self.cache.append(a)
        return a

    def _backward(self, y):
        """역전파: 손실(MSE + L2)의 기울기를 연쇄법칙으로 출력층부터 계산"""
        n = y.shape[0]
        dA = 2 * (self.cache[-1] - y) / n
        gW, gB = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            if i != len(self.W) - 1:
                dA = dA * (self.cache[i + 1] > 0)
            gW[i] = self.cache[i].T @ dA + self.l2 * self.W[i]
            gB[i] = dA.sum(0, keepdims=True)
            dA = dA @ self.W[i].T
        return gW, gB

    def _step(self, gW, gB, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        """Adam 최적화"""
        self.t += 1
        for i in range(len(self.W)):
            for P, G, M, V in ((self.W, gW, self.mW, self.vW), (self.B, gB, self.mB, self.vB)):
                M[i] = b1 * M[i] + (1 - b1) * G[i]
                V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                P[i] -= lr * (M[i] / (1 - b1 ** self.t)) / (np.sqrt(V[i] / (1 - b2 ** self.t)) + eps)

    def fit(self, X, y, X_val, y_val):
        """미니배치 학습 + 조기 종료. 표준화는 학습 데이터의 평균·표준편차만 사용"""
        self.xm, self.xs = X.mean(0), X.std(0)
        self.ym, self.ys = float(y.mean()), float(y.std()) or 1.0
        Xtr, ytr = (X - self.xm) / self.xs, ((y - self.ym) / self.ys).reshape(-1, 1)
        Xva, yva = (X_val - self.xm) / self.xs, ((y_val - self.ym) / self.ys).reshape(-1, 1)
        self._init(X.shape[1])

        best, best_params, wait = np.inf, None, 0
        self.best_epoch = self.epochs = 0
        for epoch in range(1, EPOCHS + 1):
            idx = self.rng.permutation(len(Xtr))
            for s in range(0, len(idx), BATCH):
                bi = idx[s:s + BATCH]
                self._forward(Xtr[bi])
                self._step(*self._backward(ytr[bi]))
            val = float(np.mean((self._forward(Xva) - yva) ** 2))
            self.epochs = epoch
            if val < best:
                best, wait, self.best_epoch = val, 0, epoch
                best_params = ([w.copy() for w in self.W], [b.copy() for b in self.B])
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        self.W, self.B = best_params
        return self

    def predict(self, X):
        return self._forward((X - self.xm) / self.xs).ravel() * self.ys + self.ym

    def equation(self, names):
        n_params = sum(w.size + b.size for w, b in zip(self.W, self.B))
        arch = " → ".join(str(s) for s in self.sizes)
        return (f"구조 {arch} (은닉층 ReLU), 파라미터 {n_params}개, "
                f"{self.epochs} 에폭 중 {self.best_epoch} 에폭의 가중치 사용")


# ── 전체 학습 ─────────────────────────────────────────
def fit_all(X, y, seed=SEED):
    """선형·비선형·인공신경망을 한 번에 학습하고 같은 테스트 데이터로 평가한다.

    반환: (models, metrics, skipped, sizes)
      models   {이름: 학습된 모델}
      metrics  {이름: {"val_rmse", "mae", "rmse", "r2"}}   (mae/rmse/r2는 테스트 기준)
      skipped  {이름: 건너뛴 이유}
      sizes    (학습, 검증, 테스트) 행 수
    """
    tr, va, te = split_indices(len(X), seed)
    models, metrics, skipped = {}, {}, {}
    for cls in (LinearModel, Poly2Model, PowerModel, ExpModel, MLPModel):
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                if cls is MLPModel:
                    m = cls(seed=seed).fit(X[tr], y[tr], X[va], y[va])
                else:
                    m = cls().fit(X[tr], y[tr])
                pv, pt = m.predict(X[va]), m.predict(X[te])
            if not (np.all(np.isfinite(pv)) and np.all(np.isfinite(pt))):
                raise ValueError("예측값이 발산했습니다")
        except Exception as e:  # 데이터에 맞지 않는 모델은 건너뛰고 나머지로 진행
            skipped[cls.name] = str(e)
            continue
        models[cls.name] = m
        metrics[cls.name] = {"val_rmse": rmse(y[va], pv), "mae": mae(y[te], pt),
                             "rmse": rmse(y[te], pt), "r2": r2_score(y[te], pt)}
    base = np.full(len(te), y[tr].mean())
    metrics["평균값 예측(기준선)"] = {"val_rmse": rmse(y[va], np.full(len(va), y[tr].mean())),
                                "mae": mae(y[te], base), "rmse": rmse(y[te], base),
                                "r2": r2_score(y[te], base)}
    return models, metrics, skipped, (len(tr), len(va), len(te))
