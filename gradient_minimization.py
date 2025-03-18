import numpy as np


def f1(x, y):
    return x ** 2 + y ** 2


def f2(x, y):
    return (x + 2) ** 2 + y ** 2


B = 1.5


def f3(x, y):
    return x ** 2 + B * x * y + y ** 2


def multimodal_f(x, y):
    return 20 + x ** 2 + y ** 2 - 10 * np.cos(2*np.pi * x) - 10 * np.cos(2 * np.pi * y)


def grad_f(f, x, y):
    eps = np.finfo(float).eps
    delta_x = np.sqrt(eps) * max(1.00, abs(x))
    delta_y = np.sqrt(eps) * max(1.00, abs(y))
    df_dx = (f(x + delta_x, y) - f(x - delta_x, y)) / (2 * delta_x)
    df_dy = (f(x, y + delta_y) - f(x, y - delta_y)) / (2 * delta_y)

    return np.array([df_dx, df_dy])


def goldstein(f, x, y, grad, c1, c2, a0, iterations):
    p = -grad
    a_l = 0.0
    a_r = a0
    direction = np.dot(grad, p)

    fxy = f(x, y)

    for _ in range(iterations):
        a = 0.5 * (a_r - a_l)
        x_k = x + a * p[0]
        y_k = y + a * p[1]
        l_a_c1 = fxy + c1 * a * direction
        l_a_c2 = fxy + c2 * a * direction
        func_value = f(x_k, y_k)

        if func_value > l_a_c1:
            a_r = a
        elif func_value < l_a_c2:
            a_l = a
        else:
            break

    return x_k, y_k


def armijo_gradient_descent(f, x, y, grad, c1, a0, q):
    p = -grad
    a = a0
    direction = np.dot(grad, p)

    while True:
        x_k = x + a * p[0]
        y_k = y + a * p[1]

        func_value = f(x_k, y_k)
        l_a = f(x, y) + c1 * a * direction

        if func_value <= l_a:
            break
        else:
            a *= q

    return x_k, y_k


def golden_section(f, x, y, grad, l, r, stop=1e-6):
    p = -grad
    c_k_coeff = np.float64(0.382)
    d_k_coeff = np.float64(0.618)
    a_l = l + c_k_coeff * (r - l)
    a_r = l + d_k_coeff * (r - l)
    f_l_val = f(x + a_l * p[0], y + a_l * p[1])
    f_r_val = f(x + a_r * p[0], y + a_r * p[1])
    while (r - l) > stop:
        if f_l_val > f_r_val:
            l = a_l
            a_l = a_r
            f_l_val = f_r_val
            a_r = l + d_k_coeff * (r - l)
            f_r_val = f(x + a_r * p[0], y + a_r * p[1])
        else:
            r = a_r
            a_r = a_l
            f_r_val = f_l_val
            a_l = l + c_k_coeff * (r - l)
            f_l_val = f(x + a_l * p[0], y + a_l * p[1])
    return x + (l + c_k_coeff * (r - l)) * p[0], y + (l + c_k_coeff * (r - l)) * p[1]


def get_points_for_dihotomiya(l, r):
    c_k = l + ((r - l) / 2)
    d_k = l + ((c_k - l) / 2)
    t_k = c_k + ((r - c_k) / 2)
    return c_k, d_k, t_k


def dihotomiya(f, x, y, grad, l, r, stop=1e-6):
    p = -grad
    c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
    while (r - l) > stop:
        f_c_k = f(x + c_k * p[0], y + c_k * p[1])
        f_d_k = f(x + d_k * p[0], y + d_k * p[1])
        f_t_k = f(x + t_k * p[0], y + t_k * p[1])
        if f_c_k > f_d_k:
            r = c_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
        elif f_c_k > f_t_k:
            l = c_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
        else:
            l = d_k
            r = t_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
    return x + c_k * p[0], y + c_k * p[1]


def make_step(f, x, y, h, grad, method, iteration):
    match method:
        case "default":
            return x - h * grad[0], y - h * grad[1]
        case "decreasing_lr":
            return x - (h / np.sqrt((iteration + 1))) * grad[0], y - (h / np.sqrt((iteration + 1))) * grad[1]
        case "Armijo":
            return armijo_gradient_descent(f, x, y, grad, 0.5, 1.0, 0.5)
        case "Goldstein":
            return goldstein(f, x, y, grad, 0.3, 0.7, 1.0, 3000)
        case "golden_section":
            return golden_section(f, x, y, grad, 0.0, 2.0)
        case "dihotomiya":
            return dihotomiya(f, x, y, grad, 0.0, 2.0)


def gradient_descent(f, x0, y0, h, method, iterations, stop):
    x, y = x0, y0
    for i in range(iterations):
        grad = grad_f(f, x, y)
        if np.linalg.norm(grad) < stop:
            break
        x, y = make_step(f, x, y, h, grad, method, i)

    return x, y


print("Default f1: ", gradient_descent(f1, 100, -50, 0.01, "default", 3000, 0.001))
print("Default f2: ", gradient_descent(f2, 100, -50, 0.01, "default", 3000, 0.001))
print("Armijo f2: ", gradient_descent(f2, -5, 20, 0.01, "Armijo", 3000, 0.001))
print("Armijo f3: ", gradient_descent(f3, -100, -100, 0.01, "Armijo", 3000, 0.001))
print("Default f3: ", gradient_descent(f3, -100, -100, 0.01, "default", 3000, 0.001))
print("Goldstein f2: ", gradient_descent(f2, -100, -100, 0.01, "Goldstein", 3000, 0.001))
print("Decreasing_lr f3: ", gradient_descent(f3, -100, -100, 0.01, "decreasing_lr", 300000, 0.001))
print("Decreasing_lr f2: ", gradient_descent(f2, -5, 20, 0.01, "decreasing_lr", 300000, 0.001))
print("Decreasing_lr f1: ", gradient_descent(f1, 15, 2, 0.01, "decreasing_lr", 300000, 0.001))
print("Golden_section f2: ", gradient_descent(f2, -5, -20, 0.01, "golden_section", 3000, 0.001))
print("Golden_section f3: ", gradient_descent(f3, -100, -100, 0.01, "golden_section", 3000, 0.001))
print("Golden_section f1: ", gradient_descent(f1, -100, -100, 0.01, "golden_section", 3000, 0.001))
print("Dihotomiya f2: ", gradient_descent(f2, -5, -20, 0.01, "dihotomiya", 3000, 0.001))
print("Dihotomiya f3: ", gradient_descent(f3, -100, -100, 0.01, "dihotomiya", 3000, 0.001))
print("Dihotomiya f1: ", gradient_descent(f1, -100, -100, 0.01, "dihotomiya", 3000, 0.001))
print("Dihotomiya multimodal_f: ", gradient_descent(multimodal_f, 1, 0, 0.01, "dihotomiya", 3000, 0.001))
print("Armijo multimodal_f: ", gradient_descent(multimodal_f, 1, 0, 0.01, "Armijo", 3000, 0.001))
print("Default multimodal_f: ", gradient_descent(multimodal_f, 1, 0, 0.01, "default", 3000, 0.001))
print("Decreasing_lr multimodal_f: ", gradient_descent(multimodal_f, 1, 0, 0.01, "decreasing_lr", 30000, 0.0000001))
print("Goldstein multimodal_f: ", gradient_descent(multimodal_f, 1, 0, 0.01, "Goldstein", 3000, 0.001))
print("Golden_section multimodal_f: ", gradient_descent(multimodal_f, 1, 0, 0.001, "golden_section", 300000, 0.000001))


