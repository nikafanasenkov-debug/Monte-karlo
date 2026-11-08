import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erf, sici   # только для "эталонных" значений

np.random.seed(42)


# ============================================================
# БЛОК 1. Оценка π через площадь круга
# ============================================================
def estimate_pi(n):
    """Бросаем n точек в [0,1]². Доля попавших в x²+y²≤1 → π/4."""
    x = np.random.rand(n)
    y = np.random.rand(n)
    inside = (x*x + y*y) <= 1.0
    return 4.0 * inside.mean(), x, y, inside


N_pi = 1_000_000
pi_est, x_pi, y_pi, inside_pi = estimate_pi(N_pi)

print(f"Monte-Carlo  π ≈ {pi_est:.6f}")
print(f"Точное       π = {math.pi:.6f}")
print(f"Ошибка         = {abs(pi_est - math.pi):.2e}")


# ============================================================
# БЛОК 2. Неберущиеся интегралы
# ============================================================
def mc_sample_mean(f, a, b, n):
    """∫ₐᵇ f(x) dx ≈ (b-a)·mean(f(xᵢ)) — оценка по среднему."""
    x = np.random.uniform(a, b, n)
    return (b - a) * f(x).mean()

def mc_hit_miss(f, a, b, y_max, n):
    """∫ₐᵇ f(x) dx ≈ (b-a)·y_max·(доля точек под графиком)."""
    x = np.random.uniform(a, b, n)
    y = np.random.uniform(0.0, y_max, n)
    return (b - a) * y_max * np.mean(y <= f(x))


# --- Интеграл 1: ∫₀¹ e^(−x²) dx = (√π/2)·erf(1) ≈ 0.746824132812427
f1 = lambda x: np.exp(-x*x)
exact1 = np.sqrt(np.pi) / 2 * erf(1.0)

# --- Интеграл 2: ∫₀¹ sin(x)/x dx = Si(1) ≈ 0.9460830703671831
def f2(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)                 # предел при x→0 равен 1
    m = np.abs(x) > 1e-12
    out[m] = np.sin(x[m]) / x[m]
    return out
exact2 = sici(1.0)[0]

N_int = 1_000_000

print("\n" + "=" * 62)
print("Интегралы методом Монте-Карло")
print("=" * 62)

for name, f, exact, ymax in [
    ("∫₀¹ e^(−x²) dx",       f1, exact1, 1.0),
    ("∫₀¹ sin(x)/x dx",      f2, exact2, 1.0),
]:
    est_sm = mc_sample_mean(f, 0, 1, N_int)
    est_hm = mc_hit_miss (f, 0, 1, ymax, N_int)
    print(f"\n{name}")
    print(f"  Точное значение:      {exact:.10f}")
    print(f"  Среднее (sample-mean):{est_sm:.10f}   err = {abs(est_sm-exact):.2e}")
    print(f"  Hit-or-miss:          {est_hm:.10f}   err = {abs(est_hm-exact):.2e}")


# ============================================================
# Визуализация
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# --- π: круг
a = ax[0]
k = 5000
a.scatter(x_pi[inside_pi][:k],  y_pi[inside_pi][:k],  s=1, c='green', alpha=.5, label='внутри')
a.scatter(x_pi[~inside_pi][:k], y_pi[~inside_pi][:k], s=1, c='red',   alpha=.5, label='снаружи')
t = np.linspace(0, np.pi/2, 200)
a.plot(np.cos(t), np.sin(t), 'k-', lw=2)
a.set_aspect('equal'); a.set_title(f'π ≈ {pi_est:.5f}  (истинное {math.pi:.5f})')
a.legend(markerscale=10)

# --- Интеграл 1
for ax_i, (name, f, exact, ymax) in zip(
        ax[1:],
        [("∫₀¹ e^(−x²) dx", f1, exact1, 1.0),
         ("∫₀¹ sin(x)/x dx", f2, exact2, 1.0)]):
    xs = np.random.uniform(0, 1, 5000)
    ys = np.random.uniform(0, 1, 5000)
    hit = ys <= f(xs)
    ax_i.scatter(xs[hit],  ys[hit],  s=2, c='green', alpha=.5, label='под графиком')
    ax_i.scatter(xs[~hit], ys[~hit], s=2, c='red',   alpha=.5, label='над графиком')
    xx = np.linspace(0, 1, 400)
    ax_i.plot(xx, f(xx), 'k-', lw=2)
    est = mc_hit_miss(f, 0, 1, ymax, N_int)
    ax_i.set_title(f'{name} ≈ {est:.5f}  (точное {exact:.5f})')
    ax_i.legend(markerscale=5)

plt.tight_layout()
plt.savefig('monte_carlo.png', dpi=110)
plt.show()