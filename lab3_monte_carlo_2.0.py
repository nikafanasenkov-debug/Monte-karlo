from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm' 

from scipy.integrate import quad
from scipy.stats import norm

Z = float(norm.ppf(0.975))
METHODS = ('hit_or_miss', 'mean_value')
LABELS = {'hit_or_miss': 'Попадания', 'mean_value': 'Среднее'}
COLORS = {'hit_or_miss': 'tab:orange', 'mean_value': 'tab:blue'}


def sinc_unscaled(x):
    return np.sinc(np.asarray(x, dtype=float) / np.pi)

def log_ratio(x):
    """log(1+x)/x: log1p и деление без вычисления 0/0."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    np.divide(np.log1p(x), x, out=out, where=(x != 0))
    return out

def tasks():
    return [
        dict(key='pi', title='Число π', f=lambda x: np.sqrt(1-x*x), scale=4., ref=math.pi, quad_err=0.),
        # ИСПРАВЛЕНО: \; добавляет отступ после интеграла, \, добавляет отступ перед dx
        dict(key='gauss', title=r'$\underset{0}{\overset{1}{\int}}\; e^{-x^2} \, dx$', f=lambda x: np.exp(-x*x), scale=1.),
        dict(key='sinc', title=r'$\underset{0}{\overset{1}{\int}}\; \frac{\sin(x)}{x} \, dx$', f=sinc_unscaled, scale=1.),
        dict(key='log', title=r'$\underset{0}{\overset{1}{\int}}\; \frac{\ln(1+x)}{x} \, dx$', f=log_ratio, scale=1.),
        dict(key='cos', title=r'$\underset{0}{\overset{1}{\int}}\; \cos(x^2) \, dx$', f=lambda x: np.cos(x*x), scale=1.),
    ]

def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def estimate(task, method, n, rng, keep=False):
    start = perf_counter()
    is_pi = task.get('key') == 'pi'
    x = rng.random(n)
    if is_pi:
        x = 2*x-1
    values = np.asarray(task['f'](x), dtype=float)
    if values.shape != x.shape or not np.all(np.isfinite(values)):
        raise ValueError('Функция должна возвращать конечный массив формы x.')
    if method == 'hit_or_miss':
        if np.any((values < 0) | (values > 1)):
            raise ValueError('Для выбранного прямоугольника требуется 0 <= f <= 1.')
        y = rng.random(n)
        if is_pi:
            y = 2*y-1
            observations = (x*x + y*y <= 1.).astype(float)
        else:
            observations = (y <= values).astype(float)
    else:
        y = None
        observations = values
    scale = task['scale']
    value = float(scale * observations.mean())
    se = float(scale * observations.std(ddof=1) / np.sqrt(n))
    elapsed = perf_counter() - start
    result = dict(estimate=value, se=se, ci_low=value-Z*se,
                  ci_high=value+Z*se, seconds=elapsed)
    if keep:
        result.update(x=x, y=y, observations=observations)
    return result

def validate_implementation():
    checks = []
    def check(name, condition):
        checks.append({'test': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)
    check('sin(x)/x в нуле', np.isclose(sinc_unscaled(0.), 1.))
    check('ln(1+x)/x в нуле', np.isclose(log_ratio(0.), 1.))
    check('Векторная обработка нуля', np.allclose(log_ratio(np.array([0., 1.])), [1., np.log(2.)]))
    const = dict(f=lambda x: np.ones_like(x), scale=1.)
    for method in METHODS:
        r = estimate(const, method, 100, np.random.default_rng(11))
        check('Интеграл единицы: ' + method, r['estimate'] == 1. and r['se'] == 0.)
    t = dict(f=lambda x: x, scale=1.)
    a = estimate(t, 'mean_value', 1000, np.random.default_rng(12))
    b = estimate(t, 'mean_value', 1000, np.random.default_rng(12))
    check('Воспроизводимость seed', a['estimate'] == b['estimate'] and a['se'] == b['se'])
    for f, target, name in [(lambda x: x, .5, 'x'), (lambda x: x*x, 1/3, 'x²')]:
        value, error = quad(f, 0, 1, epsabs=1e-12, epsrel=1e-12)
        check('Эталонная квадратура: ' + name, abs(value-target) < 1e-11)
    return checks

def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def geometry_plot(task, result, out, n_plot):
    count = min(n_plot, len(result['x']))
    x, y = result['x'][:count], result['y'][:count]
    inside = result['observations'][:count].astype(bool)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[inside], y[inside], s=3, alpha=.35, color='tab:green', label='Попадание', rasterized=True)
    ax.scatter(x[~inside], y[~inside], s=3, alpha=.35, color='tab:red', label='Промах', rasterized=True)
    if task['key'] == 'pi':
        angle = np.linspace(0, 2*np.pi, 600)
        ax.plot(np.cos(angle), np.sin(angle), 'k', lw=2, label='Единичная окружность')
        ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], 'k--', lw=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    else:
        xx = np.linspace(0, 1, 500)
        ax.plot(xx, task['f'](xx), 'k', lw=2, label='Граница области')
        ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k--', lw=1)
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.02, 1.02)
    ax.set(xlabel='x', ylabel='y',
           title=f"{task['title']}\nОценка: {result['estimate']:.7f}; показано {count} точек")
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=9)
    save_figure(fig, out / f"01_geometry_{task['key']}.png")

def convergence_plot(task, trajectories, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, r in trajectories.items():
        z = r['observations']
        ns = np.unique(np.geomspace(100, len(z), 150).astype(int))
        sums, squares = np.cumsum(z), np.cumsum(z*z)
        m = sums[ns-1] / ns
        variance = np.maximum((squares[ns-1] - ns*m*m)/(ns-1), 0.)
        means = task['scale'] * m
        se = task['scale'] * np.sqrt(variance/ns)
        ax.plot(ns, means, color=COLORS[method], label=LABELS[method])
        ax.fill_between(ns, means-Z*se, means+Z*se, color=COLORS[method], alpha=.15)
    ax.axhline(task['ref'], color='black', ls='--', label='Эталон')
    ax.set(xscale='log', xlabel='N', ylabel='Оценка', title=task['title'] + ': одна траектория')
    ax.legend()
    ax.grid(alpha=.25)
    save_figure(fig, out / f"02_convergence_{task['key']}.png")

def research_plot(task, rows, raw, out):
    # Каждая диагностика сохраняется отдельно: один график на PNG-файл.
    fig_rmse, ax_rmse = plt.subplots(figsize=(8, 5))
    fig_coverage, ax_coverage = plt.subplots(figsize=(8, 5))
    fig_errors, ax_errors = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        subset = [r for r in rows if r['method'] == method]
        ns = np.array([r['n'] for r in subset])
        rmse = np.array([r['rmse'] for r in subset])
        color = COLORS[method]
        ax_rmse.loglog(ns, rmse, 'o-', color=color, label=LABELS[method])
        ax_rmse.loglog(ns, rmse[0]*np.sqrt(ns[0]/ns), '--', color=color, alpha=.6)
        ax_coverage.semilogx(ns, [r['coverage'] for r in subset], 'o-', color=color, label=LABELS[method])
        vals = raw[(method, int(ns[-1]))]
        se = np.array([v['se'] for v in vals])
        errs = np.array([v['estimate']-task['ref'] for v in vals])
        zz = errs[se>0]/se[se>0]
        ax_errors.hist(zz, bins=12, density=True, alpha=.4, color=color, label=LABELS[method])
    
    ax_rmse.set(xlabel='N', ylabel='RMSE', title=task['title'] + '\nСходимость; пунктир: $N^{-1/2}$')
    ax_coverage.axhline(.95, ls='--', color='black')
    ax_coverage.set(xlabel='N', ylabel='Доля покрытия', ylim=(0,1.02),
                    title=task['title'] + '\nПокрытие 95%-интервалами')
    xx = np.linspace(-4,4,300)
    ax_errors.plot(xx, norm.pdf(xx), 'k--', label='N(0,1)')
    ax_errors.set(xlabel='(оценка − эталон)/SE', ylabel='Плотность',
                  title=task['title'] + '\nСтандартизованные ошибки')
    for ax in (ax_rmse, ax_coverage, ax_errors):
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save_figure(fig_rmse, out / f"03_rmse_{task['key']}.png")
    save_figure(fig_coverage, out / f"04_coverage_{task['key']}.png")
    save_figure(fig_errors, out / f"05_errors_{task['key']}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=1_000_000)
    parser.add_argument('--repeats', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot-points', type=int, default=5000)
    parser.add_argument('--out', type=Path, default=Path('results_mc'))
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.quick:
        args.n, args.repeats = 30_000, 15
    if args.n < 100 or args.repeats < 2 or args.plot_points < 1:
        parser.error('Нужны n >= 100, repeats >= 2, plot-points >= 1.')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checks = validate_implementation()
    save_csv(out/'validation.csv', checks)
    # Раздельные воспроизводимые потоки для основного и повторного экспериментов.
    main_seed, research_seed = np.random.SeedSequence(args.seed).spawn(2)
    main_rng = np.random.default_rng(main_seed)
    research_rng = np.random.default_rng(research_seed)
    problem_list = tasks()
    ns = np.unique(np.geomspace(100, min(args.n, 100_000), 7).astype(int))
    summary, research, trials, conclusions = [], [], [], []
    for task in problem_list:
        if 'ref' not in task:
            task['ref'], task['quad_err'] = quad(lambda x: float(task['f'](x)), 0, 1,
                                                epsabs=1e-12, epsrel=1e-12)
        # Теоретические дисперсии единичного наблюдения для сравнения с экспериментом.
        integral = task['ref']/task['scale']
        # Для pi X равномерен на [-1,1], а f и f² четные:
        # E[f(X)²] = (1/2)*integral_{-1}^{1} f(x)² dx = integral_0^1 f(x)² dx.
        second, _ = quad(lambda x: float(task['f'](x))**2, 0, 1,
                         epsabs=1e-12, epsrel=1e-12)
        variances = {'hit_or_miss': task['scale']**2*integral*(1-integral),
                     'mean_value': task['scale']**2*max(second-integral**2, 0.)}
        trajectories = {}
        for method in METHODS:
            r = estimate(task, method, args.n, main_rng, keep=True)
            trajectories[method] = r
            summary.append(dict(task=task['key'], method=method, n=args.n,
                reference=task['ref'], quad_error_estimate=task['quad_err'],
                estimate=r['estimate'], absolute_error=abs(r['estimate']-task['ref']),
                relative_error=abs(r['estimate']-task['ref'])/abs(task['ref']),
                se=r['se'], ci_low=r['ci_low'], ci_high=r['ci_high'],
                covered=r['ci_low'] <= task['ref'] <= r['ci_high'], seconds=r['seconds']))
        geometry_plot(task, trajectories['hit_or_miss'], out, args.plot_points)
        convergence_plot(task, trajectories, out)
        del trajectories
        raw, local = {}, []
        for method in METHODS:
            for n in ns:
                n = int(n)
                # Каждый повтор и каждый N используют новый участок случайного потока.
                reps = [estimate(task, method, n, research_rng) for _ in range(args.repeats)]
                raw[(method,n)] = reps
                values = np.array([r['estimate'] for r in reps])
                errors = values-task['ref']
                covered = np.array([r['ci_low'] <= task['ref'] <= r['ci_high'] for r in reps])
                empirical_sd = float(values.std(ddof=1))
                theory_sd = math.sqrt(variances[method]/n)
                row = dict(task=task['key'], method=method, n=n, repeats=args.repeats,
                    bias=float(errors.mean()), mae=float(np.abs(errors).mean()),
                    rmse=float(np.sqrt(np.mean(errors**2))), empirical_sd=empirical_sd,
                    theoretical_sd=theory_sd, sd_ratio=empirical_sd/theory_sd,
                    mean_se=float(np.mean([r['se'] for r in reps])), coverage=float(covered.mean()),
                    mean_ci_width=float(np.mean([r['ci_high']-r['ci_low'] for r in reps])),
                    mean_seconds=float(np.mean([r['seconds'] for r in reps])))
                research.append(row)
                local.append(row)
                for repeat, r in enumerate(reps, 1):
                    trials.append(dict(task=task['key'], method=method, n=n, repeat=repeat,
                        **r, error=r['estimate']-task['ref'],
                        covered=r['ci_low'] <= task['ref'] <= r['ci_high']))
            subset = [r for r in local if r['method'] == method]
            slope = float(np.polyfit(np.log([r['n'] for r in subset]),
                                     np.log([r['rmse'] for r in subset]), 1)[0])
            last = subset[-1]
            conclusions.append(f"{task['title']}; {LABELS[method]}: наклон log(RMSE)/log(N) = {slope:.3f}; "
                f"при N={last['n']}: RMSE={last['rmse']:.6g}, покрытие={last['coverage']:.3f}, "
                f"SD_эксп/SD_теор={last['sd_ratio']:.3f}.")
        research_plot(task, local, raw, out)
        hm = [r for r in local if r['method']=='hit_or_miss'][-1]
        mv = [r for r in local if r['method']=='mean_value'][-1]
        conclusions.append(f"{task['title']}: отношение эмпирических дисперсий HM/среднее = "
                           f"{(hm['empirical_sd']/mv['empirical_sd'])**2:.3f}; "
                           f"теоретическое отношение = {variances['hit_or_miss']/variances['mean_value']:.3f}.")
    save_csv(out/'summary.csv', summary)
    save_csv(out/'research.csv', research)
    save_csv(out/'trials.csv', trials)
    coverage_noise = math.sqrt(.95*.05/args.repeats)
    for r in summary:
        report.append(f"- {r['task']}, {LABELS[r['method']]}: {r['estimate']:.9f}; "
                      f"эталон {r['reference']:.9f}; |ошибка|={r['absolute_error']:.4g}; "
                      f"95% ДИ [{r['ci_low']:.9f}, {r['ci_high']:.9f}].")
    (out/'report.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))
    print(f'\nРезультаты: {out.resolve()}')
    if args.show:
        # Графики сначала сохранены через Agg; показываем их в доступном GUI-бэкенде.
        try:
            plt.switch_backend('TkAgg')
            for path in sorted(out.glob('*.png')):
                fig, ax = plt.subplots(figsize=(10,6))
                ax.imshow(plt.imread(path))
                ax.axis('off')
                fig.suptitle(path.name)
            plt.show()
        except (ImportError, RuntimeError) as exc:
            print(f'Окна недоступны ({exc}); сохраненные PNG можно открыть вручную.')

if __name__ == '__main__':
    main()
