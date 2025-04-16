import numpy as np
import optuna

# Предполагается, что модули gradient_minimization и functions
# содержат реализации функций gradient_descent и целевых функций соответственно.
from gradient_minimization import gradient_descent  # ваша реализация градиентного спуска
from functions import (
    f1, f1_1, f1_2, f1_3,
    ackley_f, multimodal_f, noisy_multimodal_f
)


def optimize_for_function(f, func_name, n_trials=50):
    def objective(trial):
        h = trial.suggest_loguniform("h", 1e-4, 1.0)
        newton_h = trial.suggest_uniform("newton_h", 0.1, 3.0)
        iterations = trial.suggest_int("iterations", 1000, 50000)
        coords = gradient_descent(
            f,
            x0=5,
            y0=2,
            method="decreasing_lr",
            h=h,
            newton_h=newton_h,
            iterations=iterations
        )
        x_opt, y_opt = coords[0], coords[1]
        # Целевая функция f имеет минимум 0, чем ближе значение к 0, тем лучше.
        f_val = f([x_opt, y_opt])
        return f_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Выводим результаты оптимизации для текущей функции
    print("=" * 70)
    print(f"Функция: {func_name}")
    print("Оптимизация гиперпараметров методом 'decreasing_lr':")
    print("-" * 70)
    print("Лучшие гиперпараметры:")
    print(f"  h         = {study.best_trial.params['h']:.6f}")
    print(f"  newton_h  = {study.best_trial.params['newton_h']:.6f}")
    print(f"  iterations= {study.best_trial.params['iterations']}")
    print(f"Значение целевой функции: {study.best_trial.value:.6e}")
    print("-" * 70)
    print("Детализация испытаний:")
    for trial in study.trials:
        print(f"  Trial {trial.number:2d}: Value = {trial.value:.6e}, Params = {trial.params}")
    print("=" * 70 + "\n")


def main():
    # Список целевых функций с описанием.
    functions = [
        (f1, "f1 = x^2 + y^2"),
        (f1_1, "f1_1 = (x + 2)^2 + y^2"),
        (f1_2, "f1_2 = x^2 + (y - 3)^2"),
        (f1_3, "f1_3 = (x - 2)^2 + (y + 1)^2"),
        (ackley_f, "ackley_f"),
        (multimodal_f, "multimodal_f"),
        (noisy_multimodal_f, "noisy_multimodal_f")
    ]
    for func, name in functions:
        optimize_for_function(func, name, n_trials=50)


if __name__ == '__main__':
    main()
