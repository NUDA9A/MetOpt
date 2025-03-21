from functions import np
from scipy.optimize import line_search


def grad_f(f, x, y):
    eps = np.finfo(float).eps
    delta_x = np.sqrt(eps) * max(1.00, abs(x))
    delta_y = np.sqrt(eps) * max(1.00, abs(y))
    df_dx = (f([x + delta_x, y]) - f([x - delta_x, y])) / (2 * delta_x)
    df_dy = (f([x, y + delta_y]) - f([x, y - delta_y])) / (2 * delta_y)

    return np.array([df_dx, df_dy])


def goldstein(f, x, y, grad, c1, c2, a0, iterations, log_file):
    p = -grad
    a_l = 0.0
    a_r = a0
    direction = np.dot(grad, p)

    fxy = f([x, y])

    for _ in range(iterations):
        a = 0.5 * (a_r - a_l)
        x_k = x + a * p[0]
        y_k = y + a * p[1]
        log_file.write(str(x_k) + ' ' + str(y_k) + '\n')
        l_a_c1 = fxy + c1 * a * direction
        l_a_c2 = fxy + c2 * a * direction
        func_value = f([x_k, y_k])

        if func_value > l_a_c1:
            a_r = a
        elif func_value < l_a_c2:
            a_l = a
        else:
            break

    return x_k, y_k


def armijo_gradient_descent(f, x, y, grad, c1, a0, q, log_file):
    p = -grad
    a = a0
    direction = np.dot(grad, p)

    while True:
        x_k = x + a * p[0]
        y_k = y + a * p[1]
        log_file.write(str(x_k) + ' ' + str(y_k) + '\n')

        func_value = f([x_k, y_k])
        l_a = f([x, y]) + c1 * a * direction

        if func_value <= l_a:
            break
        else:
            a *= q

    return x_k, y_k


def golden_section(f, x, y, grad, l, r, log_file, stop=np.finfo(float).eps):
    p = -grad
    c_k_coeff = np.float64(0.382)
    d_k_coeff = np.float64(0.618)
    a_l = l + c_k_coeff * (r - l)
    a_r = l + d_k_coeff * (r - l)
    f_l_val = f([x + a_l * p[0], y + a_l * p[1]])
    f_r_val = f([x + a_r * p[0], y + a_r * p[1]])
    while (r - l) > stop:
        log_file.write(f"{x + (l + c_k_coeff * (r - l)) * p[0]} {y + (l + c_k_coeff * (r - l)) * p[1]}\n")
        if f_l_val > f_r_val:
            l = a_l
            a_l = a_r
            f_l_val = f_r_val
            a_r = l + d_k_coeff * (r - l)
            f_r_val = f([x + a_r * p[0], y + a_r * p[1]])
        else:
            r = a_r
            a_r = a_l
            f_r_val = f_l_val
            a_l = l + c_k_coeff * (r - l)
            f_l_val = f([x + a_l * p[0], y + a_l * p[1]])
    return l + c_k_coeff * (r - l)


def get_points_for_dihotomiya(l, r):
    c_k = l + ((r - l) / 2)
    d_k = l + ((c_k - l) / 2)
    t_k = c_k + ((r - c_k) / 2)
    return c_k, d_k, t_k


def dihotomiya(f, x, y, grad, l, r, log_file, stop=np.finfo(float).eps):
    p = -grad
    c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
    while (r - l) > stop:
        log_file.write(f"{x + c_k * p[0]} {y + c_k * p[1]}\n")
        f_c_k = f([x + c_k * p[0], y + c_k * p[1]])
        f_d_k = f([x + d_k * p[0], y + d_k * p[1]])
        f_t_k = f([x + t_k * p[0], y + t_k * p[1]])
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
    return c_k


def make_step(f, x, y, h, grad, method, iteration, log_file):
    match method:
        case "default":
            return x - h * grad[0], y - h * grad[1]
        case "decreasing_lr":
            return x - (h / np.sqrt((iteration + 1))) * grad[0], y - (h / np.sqrt((iteration + 1))) * grad[1]
        case "Armijo":
            return armijo_gradient_descent(f, x, y, grad, 0.5, 1.0, 0.5, log_file)
        case "Goldstein":
            return goldstein(f, x, y, grad, 0.3, 0.7, 1.0, 100, log_file)
        case "golden_section":
            return golden_section(f, x, y, grad, 0.0, 2.0, log_file)
        case "dihotomiya":
            return dihotomiya(f, x, y, grad, 0.0, 2.0, log_file)


def gradient_descent(f, x0, y0, method="default", h=0.01, iterations=2000, stop=np.finfo(float).eps):
    x, y = x0, y0
    l_s_x = x0
    l_s_y = y0
    flag = True
    with open(f.__name__ + "_" + method + ".txt", "w") as log_file:
        for i in range(iterations):
            log_file.write(f"{x} {y}" + "\n")
            grad = grad_f(f, x, y)
            if np.linalg.norm(grad) < stop:
                break
            if method == "dihotomiya" or method == "golden_section":
                if flag:
                    alpha = line_search(f=f, myfprime=lambda args: grad_f(f, args[0], args[1]), xk=np.array([x, y]), pk=-grad, maxiter=iterations)
                    if alpha[0] is None:
                        print("Can't minimize this function with scipy.optimize.line_search. The line search did not converge.")
                        flag = False
                        continue
                    l_s_x = l_s_x - alpha[0] * grad[0]
                    l_s_y = l_s_y - alpha[0] * grad[1]
                a = make_step(f, x, y, h, grad, method, i, log_file)
                x = x - a * grad[0]
                y = y - a * grad[1]
            else:
                x, y = make_step(f, x, y, h, grad, method, i, log_file)

    return [x, y, l_s_x, l_s_y]
