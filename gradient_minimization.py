from functions import np
from scipy.optimize import line_search, minimize_scalar

def hessp_factory(f):
    def hessp(x, p):
        eps = np.sqrt(np.finfo(float).eps)
        grad1 = grad_f(f, x[0] + eps * p[0], x[1] + eps * p[1])
        grad2 = grad_f(f, x[0], x[1])
        return (grad1 - grad2) / eps

    return hessp


def grad_f(f, x, y):
    eps = np.finfo(float).eps
    delta_x = np.sqrt(eps) * max(1.00, abs(x))
    delta_y = np.sqrt(eps) * max(1.00, abs(y))
    df_dx = (f([x + delta_x, y]) - f([x - delta_x, y])) / (2 * delta_x)
    df_dy = (f([x, y + delta_y]) - f([x, y - delta_y])) / (2 * delta_y)

    return np.array([df_dx, df_dy])


def hess_f(f, x, y):
    eps = np.finfo(float).eps
    delta_x = np.sqrt(eps) * max(1.00, abs(x))
    delta_y = np.sqrt(eps) * max(1.00, abs(y))

    grad_x_plus = grad_f(f, x + delta_x, y)
    grad_x_minus = grad_f(f, x - delta_x, y)
    grad_y_plus = grad_f(f, x, y + delta_y)
    grad_y_minus = grad_f(f, x, y - delta_y)

    dgrad_dx = (grad_x_plus - grad_x_minus) / (2 * delta_x)
    dgrad_dy = (grad_y_plus - grad_y_minus) / (2 * delta_y)

    H = np.array([
        [dgrad_dx[0], dgrad_dy[0]],
        [dgrad_dx[1], dgrad_dy[1]],
    ])

    return H


def goldstein(f, x, y, p, direction, c1, c2, a0, iterations, log_file, counters):
    a_l = 0.0
    a_r = a0

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


def armijo_gradient_descent(f, x, y, p, direction, c1, a0, q, log_file, counters):
    a = a0
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
    try:
        a, b, c = np.linalg.solve(A, phis)
    except np.linalg.LinAlgError:
        return alphas[np.argmin(phis)]

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
    f_c_k = f([x - c_k * grad[0], y - c_k * grad[1]])
    counters[0] += 1
    while (r - l) > stop:
        counters[2] += 1
        f_d_k = f([x - d_k * grad[0], y - d_k * grad[1]])
        counters[0] += 1
        if f_c_k > f_d_k:
            r = c_k
            f_c_k = f_d_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
            continue

        f_t_k = f([x - t_k * grad[0], y - t_k * grad[1]])
        counters[0] += 1

        if f_c_k > f_t_k:
            l = c_k
            f_c_k = f_t_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
        else:
            l = d_k
            r = t_k
            c_k, d_k, t_k = get_points_for_dihotomiya(l, r)
    return c_k


def l_search(f, x, y, grad, a_0, c1, c2):
    alpha = line_search(
        f=f,
        myfprime=lambda args: grad_f(f, args[0], args[1]),
        xk=np.array([x, y]),
        pk=-grad,
        amax=a_0,
        c1=c1,
        c2=c2,
    )

    if alpha[0] is None:
        return -1, -1

    return x - alpha[0] * grad[0], y - alpha[0] * grad[1]


def s_minimize(f, x, y, grad, method):
    alpha = minimize_scalar(
        lambda a: f([x - a * grad[0], y - a * grad[1]]),
        method=method,
    )

    return x - alpha.x * grad[0], y - alpha.x * grad[1]


def get_p_for_newton(f, x, y, grad, counters):
    H = hess_f(f, x, y)
    counters[0] += 16
    counters[1] += 4
    try:
        p = -np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        print("Гессиан вырожден")
        p = -grad

    return p


def bfgs(
    f,
    x, y,
    B,
    prev_args,
    prev_grad,
    grad,
    iteration,
    log_file,
    counters,
    c1, a0, q
):
    if iteration == 0:
        B = np.eye(2)
        prev_args = np.array([x, y])
        prev_grad = grad
        p = -grad / (np.linalg.norm(grad) + 1e-12)
    else:
        x_prev = prev_args
        s = np.array([x, y]) - x_prev
        y_vec = grad - prev_grad
        Bs = B.dot(s)

        ys = y_vec @ s
        sBs = s @ Bs

        if ys > 1e-12 and sBs > 1e-12:
            B += np.outer(y_vec, y_vec)/ys - np.outer(Bs, Bs)/sBs

        prev_args = np.array([x, y])
        prev_grad = grad

        p = -B.dot(grad)

    direction = np.dot(grad, p)
    x_k, y_k = armijo_gradient_descent(
        f=f,
        x=x,
        y=y,
        p=p,
        direction=direction,
        c1=c1,
        a0=a0,
        q=q,
        log_file=log_file,
        counters=counters
    )

    s2 = np.array([x_k - x, y_k - y])
    grad_new = grad_f(f, x_k, y_k)
    y_vec2 = grad_new - grad
    Bs2 = B.dot(s2)

    ys2 = y_vec2 @ s2
    sBs2 = s2 @ Bs2

    if ys2 > 1e-12 and sBs2 > 1e-12:
        B += np.outer(y_vec2, y_vec2)/ys2 - np.outer(Bs2, Bs2)/sBs2

    log_file.write(f"{x_k} {y_k}\n")

    return x_k, y_k, -1, -1, B, np.array([x_k, y_k]), grad_new


def make_step(
        f, x, y, h, newton_h,
        l_s_x,
        l_s_y,
        grad,
        grad_l_s,
        method,
        iteration,
        log_file,
        c1, c2, q,
        a_0,
        stop,
        counters,
        B, prev_args, prev_grad
):
    if method == "default":
        return x - h * grad[0], y - h * grad[1], -1, -1, None, None, None
    elif method == "decreasing_lr":
        return x - (h / np.sqrt((iteration + 1))) * grad[0], y - (h / np.sqrt((iteration + 1))) * grad[
            1], -1, -1, None, None, None
    elif method == "Armijo":
        a_x, a_y = armijo_gradient_descent(f, x, y, -grad, np.dot(grad, -grad), c1, a_0, q, log_file, counters)
        ls_x, ls_y = l_search(f, l_s_x, l_s_y, grad_l_s, a_0, c1, 0.9)
        return a_x, a_y, ls_x, ls_y, None, None, None
    elif method == "Goldstein":
        g_x, g_y = goldstein(f, x, y, -grad, np.dot(grad, -grad), c1, c2, a_0, 100, log_file, counters)
        ls_x, ls_y = l_search(f, l_s_x, l_s_y, grad_l_s, a_0, c1, c2)
        return g_x, g_y, ls_x, ls_y, None, None, None
    elif method == "golden_section":
        _, a2, _ = golden_section(f, x, y, grad, 0.0, a_0, counters, stop)
        sm_x, sm_y = s_minimize(f, l_s_x, l_s_y, grad_l_s, "golden")
        return x - a2 * grad[0], y - a2 * grad[1], sm_x, sm_y, None, None, None
    elif method == "dihotomiya":
        alpha = dihotomiya(f, x, y, grad, 0.0, a_0, counters, stop)
        return x - alpha * grad[0], y - alpha * grad[1], -1, -1, None, None, None
    elif method == "parabolic":
        a1, a2, a3 = golden_section(f, x, y, grad, 0.0, a_0, counters, stop)
        alpha = parabolic(f, x, y, grad, [a1, a2, a3])
        counters[0] += 3
        counters[2] += 1
        s_x, s_y = s_minimize(f, l_s_x, l_s_y, grad_l_s, "brent")
        return x - alpha * grad[0], y - alpha * grad[1], s_x, s_y, None, None, None
    elif method == "newton":
        p = get_p_for_newton(f, x, y, grad, counters)
        return x + newton_h * p[0], y + newton_h * p[1], -1, -1, None, None, None
    elif method == "newton_armijo":
        p = get_p_for_newton(f, x, y, grad, counters)
        direction = np.dot(grad, p)
        a_x, a_y = armijo_gradient_descent(f, x, y, p, direction, c1, a_0, q, log_file, counters)
        return a_x, a_y, -1, -1, None, None, None
    elif method == "bfgs":
        return bfgs(f, x, y, B, prev_args, prev_grad, grad, iteration, log_file, counters, c1, a_0, q)
    else:
        raise ValueError(f"Unknown method: {method}")


def gradient_descent(
        f, x0, y0,
        method="default",
        h=0.01,
        newton_h=1,
        iterations=2000,
        stop=np.finfo(float).eps,
        c1=0.3, c2=0.7, q=0.5,
        a_0=2.0
):
    x, y = x0, y0
    l_s_x, l_s_y = x0, y0
    counters = [0, 0, 0]
    with open(f.__name__ + "_" + method + ".txt", "w") as log_file:
        if method == "bfgs":
            B = np.eye(2)
            prev_args = np.array([x, y])
            prev_grad = grad_f(f, x, y)
            counters[1] += 1
        else:
            B = None
            prev_args = None
            prev_grad = None
        for i in range(iterations):
            counters[2] += 1
            log_file.write(f"{x} {y}" + "\n")
            grad = grad_f(f, x, y)
            grad_l_s = grad_f(f, l_s_x, l_s_y)
            counters[1] += 1
            counters[0] += 4
            if np.linalg.norm(grad) < stop:
                break
            prev_x, prev_y = x, y
            x, y, l_s_x, l_s_y, B, prev_args, prev_grad = make_step(
                f, x, y, h, newton_h,
                l_s_x, l_s_y,
                grad,
                grad_l_s,
                method, i, log_file, c1, c2, q, a_0, stop, counters,
                B, prev_args, prev_grad
            )
            delta = np.linalg.norm([x - prev_x, y - prev_y])
            if (method == "newton" or method == "newton_armijo") and np.linalg.norm(delta) < stop:
                break

    return [x, y, l_s_x, l_s_y, counters]
