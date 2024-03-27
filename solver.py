import numpy as np


def is_matrix_correct(matrix):
    n = len(matrix)

    if matrix.shape[0] != matrix.shape[1]:
        print('The dimension does not match')
        return False

    for i in range(1, n - 1):
        if abs(matrix[i][i]) < abs(matrix[i][i - 1]) + abs(matrix[i][i + 1]):
            print(f'Sufficiency conditions are not met in the {i} line')
            return False

    for i, j in [(0, 1), (n - 1, n - 2)]:
        if abs(matrix[i][i]) < abs(matrix[i][j]):
            print('Sufficiency conditions are not met')
            return False

    if any(map(lambda x: x == 0.0, matrix.diagonal())):
        print('Zero values on the main diagonal')
        return False

    return True


def solve_eq(matrix, Cn, vec_l, vec_r, B0, f, s, C0, C1, mat_corr_l, mat_corr_r):
    if not is_matrix_correct(matrix):
        print('Error in the source data')
        return -1
    # print(Cn, B0)
    h = np.dot(Cn[vec_l:vec_r], B0) + f[vec_l:vec_r]
    h[0] = h[0] + s * C0 * ((2 * mat_corr_l + 1) % 2)
    h[-1] = h[-1] + s * C1 * ((2 * mat_corr_r + 1) % 2)

    n = len(matrix)
    sol = np.zeros((n,), dtype=float)

    # Прямой ход
    v = np.zeros((n,), dtype=float)
    u = np.zeros((n,), dtype=float)


    v[0] = matrix[0][1] / (-matrix[0][0])
    u[0] = h[0] / matrix[0][0]
    # Для i-й
    for i in range(1, n - 1):
        v[i] = matrix[i][i + 1] / (-matrix[i][i] - matrix[i][i - 1] * v[i - 1])
        u[i] = (matrix[i][i - 1] * u[i - 1] - h[i]) / (-matrix[i][i] - matrix[i][i - 1] * v[i - 1])

    # Для последней n-1-й:
    v[n - 1] = 0
    # print(matrix[n - 1][n - 2], u[n - 2], h[n - 1], matrix[n - 1][n - 1], matrix[n - 1][n - 2], v[n - 2])
    u[n - 1] = (matrix[n - 1][n - 2] * u[n - 2] - h[n - 1]) / (-matrix[n - 1][n - 1] - matrix[n - 1][n - 2] * v[n - 2])

    # Обратный ход
    sol[n - 1] = u[n - 1]
    for i in range(n - 1, 0, -1):
        sol[i - 1] = v[i - 1] * sol[i] + u[i - 1]

    return sol


def create_matrix_coeffitients(a_l, b_l, c_l, a_r, b_r, c_r, nx1, nx2, dx, dt, D, num_eq, C1, C2=None):
    s = []  # diffusion number
    mat_corr_l1 = [1] * num_eq
    mat_corr_r1 = [1] * num_eq
    mat_i_corr = [0] * num_eq # корректировочный коэффцициент к количеству иттераций в матрице
    vec_l = [0] * num_eq # корректировочный коэффцициент к вектору B на левой границе
    vec_r = [None] * num_eq # корректировочный коэффцициент к вектору B на правой границе\
    mat_corr_l = [0.5] * num_eq # корректировочный коэффцициент к виду матрицы
    mat_corr_r = [0.5] * num_eq
    C_0 = [0] * num_eq
    C_last = [0] * num_eq
    f1 = np.zeros(nx1)
    f2 = None
    if nx2 is not None:
        f2 = np.zeros(nx2)
    for i in range(num_eq):
        s.append(D[i] * dt[i] / dx[i] ** 2)

        if a_l[i] != 0:
            if i == 0:
                f1[0] = s[0] * c_l[0] * dx[0] / a_l[0]
            else:
                f2[0] = s[1] * c_l[1] * dx[1] / a_l[1]
        if a_r[i] != 0:
            if i == 0:
                f1[-1] = - s[0] * c_r[0] * dx[0] / a_r[0]
            else:
                f2[-1] = - s[1] * c_r[1] * dx[1] / a_r[1]

        if (a_l[i] == 0) and (b_l[i] != 0):
            mat_corr_l1[i] = 0
            mat_corr_l[i] = 1
            mat_i_corr[i] += -1
            vec_l[i] = 1
            a_l[i] = 1
            C_0[i] = -c_l[i] / b_l[i]
            if i == 0:
                C1[0] = -c_l[0] / b_l[0]
            elif i == 1 and C2 is not None:
                C2[0] = -c_l[1] / b_l[1]
        if (a_r[i] == 0) and (b_r[i] != 0):
            mat_corr_r1[i] = 0
            mat_corr_r[i] = 1
            mat_i_corr[i] += -1
            vec_r[i] = -1
            a_r[i] = 1
            C_last[i] = -c_r[i] / b_r[i]
            if i == 0:
                C1[-1] = -c_r[0] / b_r[0]
            elif i == 1 and C2 is not None:
                C2[-1] = -c_r[1] / b_r[1]

    return mat_corr_r, mat_corr_l, mat_i_corr, mat_corr_l1, mat_corr_r1, vec_r, vec_l, C_0, C_last, s, f1, f2, C1, C2


def create_matrix(mat_corr_r, mat_corr_l, mat_corr_l1, mat_corr_r1, mat_i_corr, nx1, nx2, dx, a_l, b_l, a_r, b_r, s):
    A2 = np.array([])
    B2 = np.array([])
    A1 = np.diagflat([-0.5 * s[0] for i in range(nx1 - 1 + mat_i_corr[0])], -1) + \
         np.diagflat([1. + mat_corr_l[0] * s[0] * (1 - mat_corr_l1[0] * (dx[0] * b_l[0] / a_l[0]))] + \
                    [1. + s[0] for i in range(nx1 - 2 + mat_i_corr[0])] + \
                    [1. + mat_corr_r[0] * s[0] * (1 + mat_corr_r1[0] * (dx[0] * b_r[0] / a_r[0]))]) + \
         np.diagflat([-0.5 * s[0] for i in range(nx1 - 1 + mat_i_corr[0])], 1)

    B1 = np.diagflat([0.5 * s[0] for i in range(nx1 - 1 + mat_i_corr[0])], -1) + \
         np.diagflat([1. - mat_corr_l[0] * s[0]  * (1 - mat_corr_l1[0] * (dx[0] * b_l[0] / a_l[0]))] + \
                     [1. - s[0] for i in range(nx1 - 2 + mat_i_corr[0])] + \
                     [1. - mat_corr_r[0] * s[0]  * (1 + mat_corr_r1[0] * (dx[0] * b_r[0] / a_r[0]))]) + \
         np.diagflat([0.5 * s[0] for i in range(nx1 - 1 + mat_i_corr[0])], 1)
    if len(s) == 2:
        A2 = np.diagflat([-0.5 * s[1] for i in range(nx2 - 1 + mat_i_corr[1])], -1) + \
            np.diagflat([1. + mat_corr_l[1] * s[1] * (1 - mat_corr_l1[1] * (dx[1] * b_l[1] / a_l[1]))] + \
                        [1. + s[1] for i in range(nx2 - 2 + mat_i_corr[1])] + \
                        [1. + mat_corr_r[1] * s[1] * (1 + mat_corr_r1[1] * (dx[1] * b_r[1] / a_r[1]))]) + \
            np.diagflat([-0.5 * s[1] for i in range(nx2 - 1 + mat_i_corr[1])], 1)

        B2 = np.diagflat([0.5 * s[1] for i in range(nx2 - 1 + mat_i_corr[1])], -1) + \
             np.diagflat([1. - mat_corr_l[1] * s[1]  * (1 - mat_corr_l1[1] * (dx[1] * b_l[1] / a_l[1]))] + \
                         [1. - s[1] for i in range(nx2 - 2 + mat_i_corr[1])] + \
                         [1. - mat_corr_r[1] * s[1]  * (1 + mat_corr_r1[1] * (dx[1] * b_r[1] / a_r[1]))]) + \
             np.diagflat([0.5 * s[1] for i in range(nx2 - 1 + mat_i_corr[1])], 1)
    return A1, B1, A2, B2


def heed_changing_bond(f2, D, s, dx, C1):
    lamb = 1.2
    c1 = lamb * D[0] * (C1[1] - C1[0]) / dx[0]
    f2[0] = s * c1 * dx[1] / D[1]
    return f2


#Выводит каждый ntout-ный график
def solver(cnst, ntout, num_eq, C1, x1, C2=None, x2=None, model_oxide=False):
    D, l0, dx, dt, a_l, b_l, c_l, a_r, b_r, c_r = cnst
    nx2 = None
    nx1 = len(x1)
    if C2 is not None and len(C2) != 0:
        nx2 = len(x2)
    nt, n_out, steps_ot = ntout
    mat_corr_r, mat_corr_l, mat_i_corr, mat_corr_l1, mat_corr_r1, vec_r, vec_l, C_0, C_last, s, f1, f2, C1, C2 = \
        create_matrix_coeffitients(a_l, b_l, c_l, a_r, b_r, c_r, nx1, nx2, dx, dt, D, num_eq, C1.copy(), C2.copy())

    A1, B1, A2, B2 = create_matrix(mat_corr_r, mat_corr_l, mat_corr_l1, mat_corr_r1, mat_i_corr, nx1, nx2, dx, a_l,
                                   b_l, a_r, b_r, s)

    Cout1 = []  # list for storing C arrays at certain time steps
    Cout2 = []
    for n in range(1, max(nt)): # time is going from second time step to last
        if any(map(lambda q: np.isnan(q), C1)) or any(map(lambda q: np.isnan(q), C2)) or any(map(lambda q: q < 0.0, C1)) \
                or any(map(lambda q: q < 0.0, C2)) or any(map(lambda q: q > 1.1, C1)) or any(map(lambda q: q > 1.1, C2)):
            print('Incorrect values during calculation')
            break
        if n < nt[0]:
            Cn1 = C1.copy()
            C1[vec_l[0]:vec_r[0]] = solve_eq(A1, Cn1, vec_l[0], vec_r[0], B1, f1, s[0], C_0[0], C_last[0], mat_corr_l[0], mat_corr_r[0])
            if n % int(nt[0] / float(n_out[0])) == 0:
                Cout1.append(C1.copy())  # numpy arrays are mutable, so we need to write out a copy of C, not C itself
                # print(n)
        if num_eq == 2 and n < nt[1]:
            if model_oxide:
                f2 = heed_changing_bond(f2.copy(), D, s[1], dx, C1.copy())
            Cn2 = C2.copy()
            C2[vec_l[1]:vec_r[1]] = solve_eq(A2, Cn2, vec_l[1], vec_r[1], B2, f2, s[1], C_0[1], C_last[1], mat_corr_l[1], mat_corr_r[1])
            if (n % int(nt[1] / float(n_out[1])) == 0) and C2 is not None:
                Cout2.append(C2.copy())
                # print(n)
    return Cout1, C1, Cout2, C2