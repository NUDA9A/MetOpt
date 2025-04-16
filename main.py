import numpy as np

from gradient_minimization import gradient_descent, grad_f, hessp_factory
from functions import f1, f1_1, f1_2, f1_3, f3, f4, ackley_f, multimodal_f, noisy_multimodal_f
from scipy.optimize import minimize

import optuna

methods = [
    # "default",
    # "decreasing_lr",
    # "Armijo",
    # "Goldstein",ls
    # "golden_section",
    # "dihotomiya",
    # "parabolic",
    "newton",
     "newton_armijo",
     "Nelder-Mid",
    "bfgs",
]
functions = [
    (f1, "f1 = x^2 + y^2"),
    (f1_1, "f1_1 = (x + 2)^2 + y^2"),
    (f1_2, "f1_2 = x^2 + (y - 3)^2"),
    (f1_3, "f1_3 = (x - 2)^2 + (y + 1)^2"),
    (f3, "f3 = x^2 + Bxy + y^2, -2 < B < 2"),
    (f4, "f4 = x**2 + y**2 * np.sin(3 * x) * np.cos(2 * y)"),
    (ackley_f, "ackley_f = -20 * np.exp(-0.2 * np.sqrt((x^2+y^2)/2)) - np.exp((cos(2*pi*x)+cos(2*pi*y))/2) + 20 + np.e"),
    (multimodal_f, "multimodal_f = 20 + x^2 + y^2 - 10*cos(2*pi*x) - 10*cos(2*pi*y)"),
    (noisy_multimodal_f, "noisy_multimodal_f = ∑ sin(m·x)*cos(m·y), m = 1..M + N(0, σ)"),
]


def scipy_newton(f):
    scipy_res = minimize(
        f,
        np.array([5, 2], dtype=float),
        method='Newton-CG',
        jac=lambda x: grad_f(f, x[0], x[1]),
        hessp=hessp_factory(f),
        options={"maxiter": 2000, "disp": False, "xtol": np.finfo(float).eps},
    )
    if not scipy_res.success:
        print("scipy KvasiNewton method failed to converge")
    else:
        print(f"scipy.optimize.minimize(..., method=\"Newton-CG\") result: x={scipy_res.x[0]}, y={scipy_res.x[1]}")


def scipy_bfgs(f):
    scipy_res = minimize(
        f,
        np.array([5, 2], dtype=float),
        method='BFGS',
        options={'maxiter': 2000, 'disp': True},
    )
    if not scipy_res.success:
        print("scipy Newton method failed to converge")
    else:
        print(f"scipy.optimize.minimize(..., method=\"BFGS\") result: x={scipy_res.x[0]}, y={scipy_res.x[1]}")


def print_result():
    for method in methods:
        print("=" * 50)
        print(f"Minimization with {method} method gradient descent:")
        print("=" * 50)
        for function in functions:
            print(function[1])
            print("=" * 50)
            if method == "Nelder-Mid":
                scipy_res = minimize(function[0], np.array([5, 2], dtype=float), method='Nelder-Mead')
                if not scipy_res.success:
                    print("scipy Nelder-Mid method failed to converge")
                else:
                    print(f"scipy.optimize.minimize(..., method=\"Nelder-Mid\") result: x={scipy_res.x[0]}, y={scipy_res.x[1]}")
                print("=" * 50)
                continue
            if method == "decreasing_lr":
                coords = gradient_descent(function[0], 5, 2, method=method, h=0.1, newton_h=1.5, iterations=50000)
            else:
                coords = gradient_descent(function[0], 5, 2, method=method, newton_h=1.5)
            if method == "Armijo" or method == "Goldstein":
                print(f"Scipy.optimize.line_search result: x={coords[2]}, y={coords[3]}")
            if method == "golden_section":
                print(f"Scipy.optimize.minimize_scalar(..., method=\"golden\") result: x={coords[2]}, y={coords[3]}")
            if method == "parabolic":
                print(f"Scipy.optimize.minimize_scalar(..., method=\"brent\") result: x={coords[2]}, y={coords[3]}")
            if method == "newton" or method == "newton_armijo":
                scipy_newton(function[0])
            if method == "bfgs":
                scipy_bfgs(function[0])
            print(f"x={coords[0]}, y={coords[1]}")
            print(f"func_count={coords[4][0]}, grad_count={coords[4][1]}, iter_count={coords[4][2]}")
            print("=" * 50)


def optuna_optimization_for_armijo(f, description, n_trials=50):
    def objective(trial):
        c1 = trial.suggest_float("c1", 0.1, 0.5)
        a0 = trial.suggest_float("a0", 0.5, 3.0)
        q = trial.suggest_float("q", 0.3, 0.8)
        iterations = trial.suggest_int("iterations", 100, 50000)
        coords = gradient_descent(f, 5, 2, method="Armijo", iterations=iterations, c1=c1, a_0=a0, q=q)
        x_opt, y_opt = coords[0], coords[1]
        return f([x_opt, y_opt])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("=" * 70)
    print("Hyperparameter optimization (Armijo) for:", description)
    print("-" * 70)
    print("Best hyperparameters:")
    print(f"  c1        = {study.best_trial.params['c1']:.6f}")
    print(f"  a0        = {study.best_trial.params['a0']:.6f}")
    print(f"  q         = {study.best_trial.params['q']:.6f}")
    print(f"  iterations= {study.best_trial.params['iterations']}")
    print(f"Best objective value: {study.best_trial.value:.6e}")
    print("-" * 70)
    print("Trial details:")
    for t in study.trials:
        print(f"  Trial {t.number:2d}: value = {t.value:.6e}, params = {t.params}")
    print("=" * 70, "\n")


def optuna_optimization_for_newton_armijo(f, description, n_trials=50):
    def objective(trial):
        newton_h = trial.suggest_float("newton_h", 0.1, 3.0)
        c1 = trial.suggest_float("c1", 0.1, 0.5)
        a0 = trial.suggest_float("a0", 0.5, 3.0)
        q = trial.suggest_float("q", 0.3, 0.8)
        iterations = trial.suggest_int("iterations", 100, 50000)
        coords = gradient_descent(f, 5, 2, method="newton_armijo", newton_h=newton_h, iterations=iterations, c1=c1, a_0=a0, q=q)
        x_opt, y_opt = coords[0], coords[1]
        return f([x_opt, y_opt])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("=" * 70)
    print("Hyperparameter optimization (newton_armijo) for:", description)
    print("-" * 70)
    print("Best hyperparameters:")
    print(f"  newton_h  = {study.best_trial.params['newton_h']:.6f}")
    print(f"  c1        = {study.best_trial.params['c1']:.6f}")
    print(f"  a0        = {study.best_trial.params['a0']:.6f}")
    print(f"  q         = {study.best_trial.params['q']:.6f}")
    print(f"  iterations= {study.best_trial.params['iterations']}")
    print(f"Best objective value: {study.best_trial.value:.6e}")
    print("-" * 70)
    print("Trial details:")
    for t in study.trials:
        print(f"  Trial {t.number:2d}: value = {t.value:.6e}, params = {t.params}")
    print("=" * 70, "\n")


def main():
    print_result()
    print("\n" + "=" * 70)
    print("Hyperparameter optimization with Armijo using optuna:")
    print("=" * 70)
    for func, desc in functions:
     optuna_optimization_for_armijo(func, desc, n_trials=50)

    print("\n" + "=" * 70)
    print("Hyperparameter optimization with newton_armijo using optuna:")
    print("=" * 70)
    for func, desc in functions:
        optuna_optimization_for_newton_armijo(func, desc, n_trials=50)


if __name__ == '__main__':
    main()
