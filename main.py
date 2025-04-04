from gradient_minimization import gradient_descent
from functions import f1, f1_1, f1_2, f1_3, f3, multimodal_f, noisy_multimodal_f

methods = ["default", "decreasing_lr", "Armijo", "Goldstein", "golden_section", "dihotomiya", "parabolic"]
functions = [(f1, "f1 = x^2 + y^2"),
             (f1_1, "f1_1 = (x + 2)^2 + y^2"),
             (f1_2, "f1_2 = x^2 + (y - 3)^2"),
             (f1_3, "f1_3 = (x - 2)^2 + (y + 1)^2"),
             (f3, "f3 = x^2 + Bxy + y^2, -2 < B < 2"),
             (multimodal_f, "multimodal_f = 20 + x^2 + y^2 - 10cos(2 * pi * x) - 10cos(2 * pi * y)"),
             (noisy_multimodal_f, "noisy_multimodal_f = ∑ sin(m·x)·cos(m·y), m = 1..M + N(0, σ)")
             ]


def print_result():
    for method in methods:
        print("=" * 50)
        print(f"Minimization with {method} method gradient descent:")
        print("=" * 50)
        for function in functions:
            print(function[1])
            print("=" * 50)
            if method == "decreasing_lr":
                coords = gradient_descent(function[0], 100, 2, method=method, h=0.1, iterations=50000)
            else:
                coords = gradient_descent(function[0], 100, 2, method=method, stop=1e-6)
            if method == "Armijo" or method == "Goldstein":
                print(f"Scipy.optimize.line_search result: x={coords[2]}, y={coords[3]}")
            if method == "golden_section":
                print(f"Scipy.optimize.minimize_scalar(..., method=\"golden\") result: x={coords[2]}, y={coords[3]}")
            if method == "parabolic":
                print(f"Scipy.optimize.minimize_scalar(..., method=\"brent\") result: x={coords[2]}, y={coords[3]}")
            print(f"x={coords[0]}, y={coords[1]}")
            print(f"func_count={coords[4][0]}, grad_count={coords[4][1]}, iter_count={coords[4][2]}")
            print("=" * 50)


print_result()
