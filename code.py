import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad  # Для автоматического расчета точных значений

# Фиксируем seed для воспроизводимости
np.random.seed(42)

N_CALC = 100_000_000  # Точек для расчета (высокая точность)
N_PLOT = 15_000      # Точек для графика (чтобы красиво рисовалось)

# ============================================================
# БЛОК 1: Оценка числа Пи через площадь круга
# ============================================================
def estimate_pi_and_plot():
    # 1. Расчет на 1 млн точек
    x_calc = np.random.rand(N_CALC)
    y_calc = np.random.rand(N_CALC)
    inside_calc = (x_calc**2 + y_calc**2) <= 1.0
    pi_est = 4.0 * np.mean(inside_calc)
    
    # 2. Выборка для графика
    x_plot = x_calc[:N_PLOT]
    y_plot = y_calc[:N_PLOT]
    inside_plot = inside_calc[:N_PLOT]

    # 3. Отрисовка
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_plot[inside_plot], y_plot[inside_plot], s=5, c='green', alpha=0.6, label='внутри круга')
    ax.scatter(x_plot[~inside_plot], y_plot[~inside_plot], s=5, c='red', alpha=0.6, label='снаружи круга')
    
    # Линия окружности
    t = np.linspace(0, np.pi/2, 200)
    ax.plot(np.cos(t), np.sin(t), 'k-', lw=2)
    
    # Пунктирная рамка области разброса
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=1.5, label='область разброса точек')

    ax.set_aspect('equal')
    ax.set_title(f'Площадь круга -> Оценка π = {pi_est:.5f}\n(Истинное значение = {math.pi:.5f})', fontsize=14, pad=15)
    
    # ИЗМЕНЕНО ЗДЕСЬ: loc='lower left'
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

    print("=== БЛОК 1: Оценка π ===")
    print(f"Monte-Carlo  π ≈ {pi_est:.6f}")
    print(f"Точное       π = {math.pi:.6f}")
    print(f"Абс. ошибка    = {abs(pi_est - math.pi):.2e}\n")


# ============================================================
# БЛОК 2: Неберущиеся интегралы (Hit-or-Miss)
# ============================================================
def plot_integral(func, name_str, latex_str):
    # 1. Расчет точного значения через scipy
    exact_val, _ = quad(func, 0, 1)
    
    # 2. Метод Монте-Карло (Hit-or-Miss) на 1 млн точек
    x_calc = np.random.uniform(0, 1, N_CALC)
    y_calc = np.random.uniform(0, 1, N_CALC)
    hit_calc = y_calc <= func(x_calc)
    integral_est = np.mean(hit_calc)
    
    # 3. Выборка для графика
    x_plot = x_calc[:N_PLOT]
    y_plot = y_calc[:N_PLOT]
    hit_plot = hit_calc[:N_PLOT]

    # 4. Отрисовка
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_plot[hit_plot], y_plot[hit_plot], s=5, c='green', alpha=0.6, label='под графиком (засчитано)')
    ax.scatter(x_plot[~hit_plot], y_plot[~hit_plot], s=5, c='red', alpha=0.6, label='над графиком')
    
    # Линия функции
    xx = np.linspace(0, 1, 400)
    ax.plot(xx, func(xx), 'k-', lw=2.5, label=f'y = {name_str}')
    
    # Пунктирная рамка области разброса
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=1.5, label='область разброса точек')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Интеграл на [0, 1]: {latex_str}\nОценка = {integral_est:.5f} (точное = {exact_val:.5f})', 
                 fontsize=13, pad=15)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    print(f"=== {latex_str} ===")
    print(f"Monte-Carlo ≈ {integral_est:.6f}")
    print(f"Точное      = {exact_val:.6f}")
    print(f"Абс. ошибка = {abs(integral_est - exact_val):.2e}\n")


# ============================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================
if __name__ == "__main__":
    # Блок 1
    estimate_pi_and_plot()

    # Блок 2: Определяем функции
    f1 = lambda x: np.exp(-x**2)
    f2 = lambda x: np.sin(x) / x if isinstance(x, (int, float)) and x == 0 else np.sinc(x / np.pi)
    f3 = lambda x: np.log(1 + x) / x if isinstance(x, (int, float)) and x == 0 else np.log(1 + x) / np.where(x == 0, 1e-12, x)
    f4 = lambda x: np.cos(x**2)

    integrals = [
        (f1, "e^(-x²)", r"$\int_{0}^{1} e^{-x^2} dx$"),
        (f2, "sin(x)/x", r"$\int_{0}^{1} \frac{\sin(x)}{x} dx$"),
        (f3, "ln(1+x)/x", r"$\int_{0}^{1} \frac{\ln(1+x)}{x} dx$"),
        (f4, "cos(x²)", r"$\int_{0}^{1} \cos(x^2) dx$")
    ]

    # Запускаем отрисовку
    for func, name, latex in integrals:
        plot_integral(func, name, latex)