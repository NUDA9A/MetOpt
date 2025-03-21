from functions import np
from scipy.optimize import line_search, minimize_scalar


def grad_f(f, x, y):
    eps = np.finfo(float).eps
    delta_x = np.sqrt(eps) * max(1.00, abs(x))
    delta_y = np.sqrt(eps) * max(1.00, abs(y))
    df_dx = (f([x + delta_x, y]) - f([x - delta_x, y])) / (2 * delta_x)
    df_dy = (f([x, y + delta_y]) - f([x, y - delta_y])) / (2 * delta_y)

    return np.array([df_dx, df_dy])


def goldstein(f, x, y, grad, c1, c2, a0, iterations, log_file, counters):
    p = -grad
    a_l = 0.0
    a_r = a0
    direction = np.dot(grad, p)

    fxy = f([x, y])
    counters[0] += 1

    for _ in range(iterations):
        counters[2] += 1
        a = 0.5 * (a_r - a_l)
        x_k = x + a * p[0]
        y_k = y + a * p[1]
        log_file.write(str(x_k) + ' ' + str(y_k) + '\n')
        l_a_c1 = fxy + c1 * a * direction
        l_a_c2 = fxy + c2 * a * direction
        func_value = f([x_k, y_k])
        counters[0] += 1

        if func_value > l_a_c1:
            a_r = a
        elif func_value < l_a_c2:
            a_l = a
        else:
            break

    return x_k, y_k


def armijo_gradient_descent(f, x, y, grad, c1, a0, q, log_file, counters):
    p = -grad
    a = a0
    direction = np.dot(grad, p)
    fxy = f([x, y])
    counters[0] += 1

    while True:
        counters[2] += 1
        x_k = x + a * p[0]
        y_k = y + a * p[1]
        log_file.write(str(x_k) + ' ' + str(y_k) + '\n')

        func_value = f([x_k, y_k])
        counters[0] += 1
        l_a = fxy + c1 * a * direction

        if func_value <= l_a:
            break
        else:
            a *= q

    return x_k, y_k


def golden_section(f, x, y, grad, l, r, counters, stop=np.finfo(float).eps):
    c_k_coeff = np.float64(0.382)
    d_k_coeff = np.float64(0.618)
    a_l = l + c_k_coeff * (r - l)
    a_r = l + d_k_coeff * (r - l)
    f_l_val = f([x - a_l * grad[0], y - a_l * grad[1]])
    f_r_val = f([x - a_r * grad[0], y - a_r * grad[1]])
    counters[0] += 2
    while (r - l) > stop:
        counters[2] += 1
        if f_l_val > f_r_val:
            l = a_l
            a_l = a_r
            f_l_val = f_r_val
            a_r = l + d_k_coeff * (r - l)
            f_r_val = f([x - a_r * grad[0], y - a_r * grad[1]])
            counters[0] += 1
        else:
            r = a_r
            a_r = a_l
            f_r_val = f_l_val
            a_l = l + c_k_coeff * (r - l)
            f_l_val = f([x - a_l * grad[0], y - a_l * grad[1]])
            counters[0] += 1
    return l, l + c_k_coeff * (r - l), r


def parabolic(f, x, y, grad, alpha_vals):
    def phi(alpha):
        return f([x - alpha * grad[0], y - alpha * grad[1]])

    alphas = np.array(alpha_vals)
    phis = np.array([phi(alpha) for alpha in alphas])

    A = np.vstack([alphas ** 2, alphas, np.ones_like(alphas)]).T
    a, b, c = np.linalg.solve(A, phis)

    if a <= 0:
        res = alphas[np.argmin(phis)]
    else:
        res = -b / (2 * a)
        if res < min(alphas) or res > max(alphas):
            res = alphas[np.argmin(phis)]

    return res


def get_points_for_dihotomiya(l, r):
    c_k = l + ((r - l) / 2)
    d_k = l + ((c_k - l) / 2)
    t_k = c_k + ((r - c_k) / 2)
    return c_k, d_k, t_k


def dihotomiya(f, x, y, grad, l, r, counters, stop=np.finfo(float).eps):
    c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
    while (r - l) > stop:
        counters[2] += 1
        f_c_k = f([x - c_k * grad[0], y - c_k * grad[1]])
        counters[0] += 1
        if f_c_k > f([x - d_k * grad[0], y - d_k * grad[1]]):
            counters[0] += 1
            r = c_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
        elif f_c_k > f([x - t_k * grad[0], y - t_k * grad[1]]):
            counters[0] += 2
            l = c_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
        else:
            counters[0] += 2
            l = d_k
            r = t_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
    return c_k


def l_search(f, x, y, grad, a_0):
    alpha = line_search(
        f=f,
        myfprime=lambda args: grad_f(f, args[0], args[1]),
        xk=np.array([x, y]),
        pk=-grad,
        amax=a_0,
    )
    return x - alpha[0] * grad[0], y - alpha[0] * grad[1]


def s_minimize(f, x, y, grad, method):
    alpha = minimize_scalar(
        lambda a: f([x - a * grad[0], y - a * grad[1]]),
        method=method,
    )

    return x - alpha.x * grad[0], y - alpha.x * grad[1]


def make_step(
        f, x, y, h,
        l_s_x,
        l_s_y,
        grad,
        grad_l_s,
        method,
        iteration,
        log_file,
        c1, c2,
        a_0,
        stop,
        counters
):
    match method:
        case "default":
            return x - h * grad[0], y - h * grad[1], -1, -1
        case "decreasing_lr":
            return x - (h / np.sqrt((iteration + 1))) * grad[0], y - (h / np.sqrt((iteration + 1))) * grad[1], -1, -1
        case "Armijo":
            a_x, a_y = armijo_gradient_descent(f, x, y, grad, c1, a_0, 0.5, log_file, counters)
            ls_x, ls_y = l_search(f, l_s_x, l_s_y, grad_l_s, a_0)
            return a_x, a_y, ls_x, ls_y
        case "Goldstein":
            g_x, g_y = goldstein(f, x, y, grad, c1, c2, a_0, 100, log_file, counters)
            ls_x, ls_y = l_search(f, l_s_x, l_s_y, grad_l_s, a_0)
            return g_x, g_y, ls_x, ls_y
        case "golden_section":
            _, a2, _ = golden_section(f, x, y, grad, 0.0, a_0, counters, stop)
            sm_x, sm_y = s_minimize(f, x, y, grad, "golden")

            return x - a2 * grad[0], y - a2 * grad[1], sm_x, sm_y
        case "dihotomiya":
            alpha = dihotomiya(f, x, y, grad, 0.0, a_0, counters, stop)
            return x - alpha * grad[0], y - alpha * grad[1], -1, -1
        case "parabolic":
            a1, a2, a3 = golden_section(f, x, y, grad, 0.0, a_0, counters, stop)
            alpha = parabolic(f, x, y, grad, [a1, a2, a3])
            counters[0] += 3
            counters[2] += 1
            s_x, s_y = s_minimize(f, x, y, grad, "brent")
            return x - alpha * grad[0], y - alpha * grad[1], s_x, s_y


def gradient_descent(
        f, x0, y0,
        method="default",
        h=0.01,
        iterations=2000,
        stop=np.finfo(float).eps,
        c1=0.3, c2=0.7,
        a_0=2.0
):
    x, y = x0, y0
    l_s_x, l_s_y = x0, y0
    flag = True
    counters = [0, 0, 0]
    with open(f.__name__ + "_" + method + ".txt", "w") as log_file:
        for i in range(iterations):
            counters[2] += 1
            log_file.write(f"{x} {y}" + "\n")
            grad = grad_f(f, x, y)
            grad_l_s = grad_f(f, l_s_x, l_s_y)
            counters[1] += 1
            counters[0] += 4
            if np.linalg.norm(grad) < stop:
                break
            x, y, l_s_x, l_s_y = make_step(
                f, x, y, h,
                l_s_x, l_s_y,
                grad,
                grad_l_s,
                method, i, log_file, c1, c2, a_0, stop, counters,
            )

    return [x, y, l_s_x, l_s_y, counters]
