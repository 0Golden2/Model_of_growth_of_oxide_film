import numpy as np
import matplotlib.pyplot as plt


def is_matrix_correct(matrix):
    n = len(matrix)

    if matrix.shape[0] != matrix.shape[1]:
        print('Не соответствует размерность')
        return False

    for i in range(1, n - 1):
        if abs(matrix[i][i]) < abs(matrix[i][i - 1]) + abs(matrix[i][i + 1]):
            print(f'Не выполняются условия достаточности в {i} строке')
            return False

    for i, j in [(0, 1), (n - 1, n - 2)]:
        if abs(matrix[i][i]) < abs(matrix[i][j]):
            print('Не выполняются условия достаточности')
            return False

    if any(map(lambda x: x == 0.0, matrix.diagonal())):
        print('Нулеые значения на главной диагонали')
        return False

    return True


def read_const_file(file_name):
    with open(file_name, 'r',  encoding='utf-8') as f:
        print("""Если хотите найти равновесное решение, введите 1, если решение через n-ое время, то введите 2.""")
        k = int(input())
        constants = []
        num_eq = int(f.readline())
        D = list(map(float, f.readline().split())) #Коэффициент диффузии
        x_max = list(map(float, f.readline().split())) #длина промежутка
        dx = list(map(float, f.readline().split())) #шаг деления промежутка
        dt = list(map(float, f.readline().split())) #шаг по времени
        a1 = list(map(float, f.readline().split()))
        b1 = list(map(float, f.readline().split()))
        c1 = list(map(float, f.readline().split()))
        a2 = list(map(float, f.readline().split()))
        b2 = list(map(float, f.readline().split()))
        c2 = list(map(float, f.readline().split()))
        # C_init = list()
        nt_out = tuple(map(int,
                          f.readline().split()))  # должен содержать полное время и кол-во графиков, которые выводятся или шаг вывода графиков
        constants.append([D, x_max, dx, dt, a1, b1, c1, a2, b2, c2])
        # constants.append(C_init)
        constants.append(nt_out)
        constants.append(k)
        constants.append(num_eq)
    return constants


def solve_eq(matrix, Cn, vec_l, vec_r, B0, f, s, C0, C1, mat_corr_l, mat_corr_r):
    if not is_matrix_correct(matrix):
        print('Ошибка в исходных данных')
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


def create_matrix_coeffitients(a1, b1, c1, a2, b2, c2, nx, dx, dt, D, num_eq, C1, C2=None):
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
    f1 = np.zeros(nx)
    f2 = np.zeros(nx)
    for i in range(num_eq):
        s.append(D[i] * dt[i] / dx[i] ** 2)

        if a1[i] != 0:
            if i == 0:
                f1[0] = s[0] * c1[0] * dx[0] / a1[0]
            else:
                f2[0] = s[1] * c1[1] * dx[1] / a1[1]
        if a2[i] != 0:
            if i == 0:
                f1[-1] = - s[0] * c2[0] * dx[0] / a2[0]
            else:
                f2[-1] = - s[1] * c2[1] * dx[1] / a2[1]

        if (a1[i] == 0) and (b1[i] != 0):
            mat_corr_l1[i] = 0
            mat_corr_l[i] = 1
            mat_i_corr[i] += -1
            vec_l[i] = 1
            a1[i] = 1
            C_0[i] = -c1[i] / b1[i]
            if i == 0:
                C1[0] = -c1[0] / b1[0]
            elif i == 1 and C2 is not None:
                C2[0] = -c1[1] / b1[1]
        if (a2[i] == 0) and (b2[i] != 0):
            mat_corr_r1[i] = 0
            mat_corr_r[i] = 1
            mat_i_corr[i] += -1
            vec_r[i] = -1
            a2[i] = 1
            C_last[i] = -c2[i] / b2[i]
            if i == 0:
                C1[-1] = -c2[0] / b2[0]
            elif i == 1 and C2 is not None:
                C2[-1] = -c2[1] / b2[1]

    return mat_corr_r, mat_corr_l, mat_i_corr, mat_corr_l1, mat_corr_r1, vec_r, vec_l, C_0, C_last, s, f1, f2, C1, C2


def create_matrix(mat_corr_r, mat_corr_l, mat_corr_l1, mat_corr_r1, mat_i_corr, nx, dx, a1, b1, a2, b2, s):
    A2 = np.array([])
    B2 = np.array([])
    A1 = np.diagflat([-0.5 * s[0] for i in range(nx - 1 + mat_i_corr[0])], -1) + \
         np.diagflat([1. + mat_corr_l[0] * s[0] * (1 - mat_corr_l1[0] * (dx[0] * b1[0] / a1[0]))] + \
                    [1. + s[0] for i in range(nx - 2 + mat_i_corr[0])] + \
                    [1. + mat_corr_r[0] * s[0] * (1 + mat_corr_r1[0] * (dx[0] * b2[0] / a2[0]))]) + \
         np.diagflat([-0.5 * s[0] for i in range(nx - 1 + mat_i_corr[0])], 1)

    B1 = np.diagflat([0.5 * s[0] for i in range(nx - 1 + mat_i_corr[0])], -1) + \
         np.diagflat([1. - mat_corr_l[0] * s[0]  * (1 - mat_corr_l1[0] * (dx[0] * b1[0] / a1[0]))] + \
                     [1. - s[0] for i in range(nx - 2 + mat_i_corr[0])] + \
                     [1. - mat_corr_r[0] * s[0]  * (1 + mat_corr_r1[0] * (dx[0] * b2[0] / a2[0]))]) + \
         np.diagflat([0.5 * s[0] for i in range(nx - 1 + mat_i_corr[0])], 1)
    if len(s) == 2:
        A2 = np.diagflat([-0.5 * s[1] for i in range(nx - 1 + mat_i_corr[1])], -1) + \
            np.diagflat([1. + mat_corr_l[1] * s[1] * (1 - mat_corr_l1[1] * (dx[1] * b1[1] / a1[1]))] + \
                        [1. + s[1] for i in range(nx - 2 + mat_i_corr[1])] + \
                        [1. + mat_corr_r[1] * s[1] * (1 + mat_corr_r1[1] * (dx[1] * b2[1] / a2[1]))]) + \
            np.diagflat([-0.5 * s[1] for i in range(nx - 1 + mat_i_corr[1])], 1)

        B2 = np.diagflat([0.5 * s[1] for i in range(nx - 1 + mat_i_corr[1])], -1) + \
             np.diagflat([1. - mat_corr_l[1] * s[1]  * (1 - mat_corr_l1[1] * (dx[1] * b1[1] / a1[1]))] + \
                         [1. - s[1] for i in range(nx - 2 + mat_i_corr[1])] + \
                         [1. - mat_corr_r[1] * s[1]  * (1 + mat_corr_r1[1] * (dx[1] * b2[1] / a2[1]))]) + \
             np.diagflat([0.5 * s[1] for i in range(nx - 1 + mat_i_corr[1])], 1)
    return A1, B1, A2, B2


#Выводит каждый ntout-ный график
def find_solution(cnst, ntout, num_eq, C1, C2=None):
    D, x_max, dx, dt, a1, b1, c1, a2, b2, c2 = cnst
    x = np.arange(0, x_max[0] + dx[0], dx[0])
    nx = len(x)
    nt, n_out = ntout
    mat_corr_r, mat_corr_l, mat_i_corr, mat_corr_l1, mat_corr_r1, vec_r, vec_l, C_0, C_last, s, f1, f2, C1, C2 = \
        create_matrix_coeffitients(a1, b1, c1, a2, b2, c2, nx, dx, dt, D, num_eq, C1.copy(), C2.copy())


    A1, B1, A2, B2 = create_matrix(mat_corr_r, mat_corr_l, mat_corr_l1, mat_corr_r1, mat_i_corr, nx, dx, a1, b1, a2, b2, s)

    Cout1 = []  # list for storing C arrays at certain time steps
    Cout2 = []
    for n in range(1, nt): # time is going from second time step to last
        Cn1 = C1.copy()
        C1[vec_l[0]:vec_r[0]] = solve_eq(A1, Cn1, vec_l[0], vec_r[0], B1, f1, s[0], C_0[0], C_last[0], mat_corr_l[0], mat_corr_r[0])
        if num_eq == 2:
            Cn2 = C2.copy()
            C2[vec_l[1]:vec_r[1]] = solve_eq(A2, Cn2, vec_l[1], vec_r[1], B2, f2, s[1], C_0[1], C_last[1], mat_corr_l[1], mat_corr_r[1])
        if n % int(nt / float(n_out)) == 0 or n == nt - 1:
            Cout1.append(C1.copy())  # numpy arrays are mutable, so we need to write out a copy of C, not C itself
            if C2 is not None:
                Cout2.append(C2.copy())
            print(n)
    return Cout1, C1, Cout2, C2


def print_plot(x, C_out, C_in):
    plt.plot(x, C_out[-1], linestyle='-.', color='red')
    plt.xlabel('Глубина образца', fontsize=12)
    plt.ylabel('Концентрация', fontsize=12)
    plt.title('Конечное распределение', fontsize=14)
    plt.show()
    fig, ax = plt.subplots()
    for C in C_out:
        if list(C) == list(C_out[-1]):
            ax.plot(x, C_out[-1], linestyle='-.', color='red', label='Конечное распределение')
        else:
            ax.plot(x, C, 'k')
    ax.plot(x, C_in, linestyle='-', color='green', label='Начальное распределение')
    plt.legend(loc='upper left')
    plt.xlabel('Глубина образца', fontsize=12)
    plt.ylabel('Концентрация', fontsize=12)
    plt.title('Crank-Nicolson scheme', fontsize=14)
    plt.show()


def write_data(k, C_out):
    with open(f"Результаты уравнения {k}.txt", "w+", encoding='utf-8') as f:
        np.savetxt(f, np.array(C_out).T, fmt='%10.10f')
    print(f"Результаты уравнения {k} успешно внесены")

# Main блок
cnst, ntout, k, num_eq = read_const_file('Константы 1.txt')
nx = len(np.arange(0, cnst[1][0] + cnst[2][0], cnst[2][0]))
C_in =  np.array([1. for i in range(0, int(nx / 2))] + [0. for j in range(int(nx / 2), nx)]) # initial condition
C_in2 =  np.array([0. for p in range(0, int(nx / 2))] + [1. for l in range(int(nx / 2), nx)])
# C_in2[-1] = 1.
C1_out, C1_last, C2_out, C2_last = find_solution(cnst, ntout, num_eq,  C_in.copy(), C_in2.copy())
C1_out.append(C1_last)
C2_out.append(C2_last)

write_data(1, C1_out)
write_data(2, C2_out)
#вывод графиков
x = np.arange(0, cnst[1][0] + cnst[2][0], cnst[2][0])
print_plot(x, C1_out, C_in)
if num_eq == 2:
    print_plot(x, C2_out, C_in2)


