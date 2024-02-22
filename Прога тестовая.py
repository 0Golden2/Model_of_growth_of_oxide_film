import numpy as np
import matplotlib.pyplot as plt
# from sympy import Matrix, solve_linear_system
# from sympy.abc import x, y, z


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


# def solve_two_first_eq(mat, h):
#     system = Matrix(( (mat[0][0], mat[0][1], mat[0][2], h[0]), (mat[1][0], mat[1][1], mat[1][2], h[1]) ))
#     return solve_linear_system(system, x, y, z)


def read_const_file(file_name):
    with open(file_name, 'r',  encoding='utf-8') as f:
        print("""Если хотите найти равновесное решение, введите 1, если решение через n-ое время, то введите 2.""")
        k = int(input())
        D = float(f.readline()) #Коэффициент диффузии
        x_max = float(f.readline()) #длина промежутка
        dx = float(f.readline()) #шаг деления промежутка
        dt = float(f.readline()) #шаг по времени
        GU_left = int(f.readline()) # ГУ на левой границе
        GU_right = int(f.readline()) # ГУ на правой границе
        n_out = None
        nt = None
        nt_out = None
        a1 = 1
        b1 = 1
        c1 = 1
        a2 = 1
        b2 = 1
        c2 = 1
        if k == 1:
            n_out = int(f.readline()) # Шаг вывода графиков
        elif k == 2:
            nt = int(f.readline()) # Полное время
            nt_out = int(f.readline()) # кол-во графиков, которые выводятся
        else:
            raise ValueError('k может принимать значение 1 или 2')
        if GU_left == 3:
            a1 = float(f.readline())
            b1 = float(f.readline())
            c1 = float(f.readline())
        if GU_right == 3:
            a2 = float(f.readline())
            b2 = float(f.readline())
            c2 = float(f.readline())
    return D, x_max, dx, dt, GU_left, GU_right, nt, nt_out, n_out, k, a1, b1, c1, a2, b2, c2


def solve_eq(matrix, h):
    if not is_matrix_correct(matrix):
        print('Ошибка в исходных данных')
        return -1

    n = len(matrix)

    # if (mat_ic == np.zeros((n,))).all():
    #     h = b
    # else:
    #     h = mat_ic.dot(b)
    # print('h=', h)

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

    # elif type_GU == 2:
    #     sol_two_first_eq = solve_two_first_eq(matrix, h)
    #     keys = [key for key in sol_two_first_eq.keys()]
    #     x, y = sol_two_first_eq[keys[0]], sol_two_first_eq[keys[1]]
    #     x1 = x.args[1].as_coeff_mul()[1][0]*x.args[1].as_coeff_mul()[0]
    #     x2 = x.args[0]
    #     y1 = y.args[1].as_coeff_mul()[1][0]*y.args[1].as_coeff_mul()[0]
    #     y2 = y.args[0]
    #     v[0] = x1 / y1
    #     u[0] = v[0] * y2 + x2
    #     v[1] = y1
    #     u[1] = y2
    #
    #     for i in range(2, n - 1):
    #         v[i] = matrix[i][i + 1] / (-matrix[i][i] - matrix[i][i - 1] * v[i - 1])
    #         u[i] = (matrix[i][i - 1] * u[i - 1] - h[i]) / (-matrix[i][i] - matrix[i][i - 1] * v[i - 1])

        # # Для последней n-1-й:
        # v[n - 1] = 0
        # # print(matrix[n - 1][n - 2], u[n - 2], h[n - 1], matrix[n - 1][n - 1], matrix[n - 1][n - 2], v[n - 2])
        # u[n - 1] = (matrix[n - 1][n - 2] * u[n - 2] - h[n - 1]) / (-matrix[n - 1][n - 1] - matrix[n - 1][n - 2] * v[n - 2])

    # Обратный ход
    sol[n - 1] = u[n - 1]
    for i in range(n - 1, 0, -1):
        sol[i - 1] = v[i - 1] * sol[i] + u[i - 1]

    return sol


#Выводит каждый ntout-ный график
def find_equilibrium_solution(dx, nx, dt, D, C, GU_left, GU_right, a1=1, b1=1, c1=1, a2=1,b2=1,c2=1, ntout=1000,):
    delta = 0.00000001
    Cout = []  # list for storing V arrays at certain time steps
    s = D * dt / dx ** 2  # diffusion number
    Cn = np.zeros(nx)
    n = 1
    mat_i_corr = 0  # корректировочный коэффцициент к количеству иттераций в матрице
    vec_l = 0   # корректировочный коэффцициент к вектору B на левой границе
    vec_r = None    # корректировочный коэффцициент к вектору B на правой границе\
    mat_corr_l = 0.5  # корректировочный коэффцициент к виду матрицы
    mat_corr_r = 0.5
    mat_corr_l3 = 0 # корректировочный коэффцициент к виду матрицы для условия 3 рода
    mat_corr_r3 = 0
    C0 = 0
    C1 = 0
    f = np.zeros(nx)
    if (a1 == 0) and (b1 != 0):
        GU_left = 1
        a1 = 1
        C[0] = -c1 / b1
        C0 = C[0]
    if (a2 == 0) and (b2 != 0):
        GU_right = 1
        a2 = 1
        C[-1] = -c2 / b1
        C1 = C[-1]
    if GU_left == 1:
        mat_corr_l = 1
        mat_i_corr += -1
        vec_l = 1
        C0 = C[0] # boundary condition on left side
    elif GU_left == 3:
        mat_corr_l3 = 1
        f[0] = s * c1 * dx / a1

    if GU_right == 1:
        mat_corr_r = 1
        mat_i_corr += -1
        vec_r = -1
        C1 = C[-1]  # boundary condition on right side
    elif GU_right == 3:
        mat_corr_r3 = 1
        f[-1] = - s * c2 * dx / a2

    A = np.diagflat([-0.5 * s for i in range(nx - 1 + mat_i_corr)], -1) + \
        np.diagflat([1. + mat_corr_l * s * (1 - mat_corr_l3 * (dx * b1 / a1))] + \
                    [1. + s for i in range(nx - 2 + mat_i_corr)] + \
                    [1. + mat_corr_r * s * (1 + (dx * b2 / a2) * mat_corr_r3)]) + \
        np.diagflat([-0.5 * s for i in range(nx - 1 + mat_i_corr)], 1)

    B1 = np.diagflat([0.5 * s for i in range(nx - 1 + mat_i_corr)], -1) + \
         np.diagflat([1. - mat_corr_l * s  * (1 - (dx * b1 / a1) * mat_corr_l3)] + \
                     [1. - s for i in range(nx - 2 + mat_i_corr)] + \
                     [1. - mat_corr_r * s  * (1 + (dx * b2 / a2) * mat_corr_r3)]) + \
         np.diagflat([0.5 * s for i in range(nx - 1 + mat_i_corr)], 1)

    print(A)
    while any(map(lambda q: np.fabs(q) > delta,
                  (np.array(Cn) - np.array(C)))):  # time is going from second time step to last
        Cn = C.copy()
        if any(map(lambda q: q > 10**(2), Cn)):
            raise ValueError('Решение не найдено')
        B = np.dot(Cn[vec_l:vec_r], B1) + f[vec_l:vec_r]
        B[0] = B[0] + 0.5 * s * (C0 + C0) * ((2 * mat_corr_l + 1) % 2)
        B[-1] = B[-1] + 0.5 * s * (C1 + C1) * ((2 * mat_corr_r + 1) % 2)
        C[vec_l:vec_r] = solve_eq(A, B)
        if n % ntout == 0:
            Cout.append(C.copy())  # numpy arrays are mutable, so we need to write out a copy of V, not V itself
            print(n)
        n += 1
        if n == 10000:
            break

    # A = np.diagflat([-0.5 * s for i in range(nx - 1)], -1) + \
    #     np.diagflat([1 + 0.5 * s] + [1. + s for i in range(nx - 2)] + [1 + 0.5 * s]) + \
    #     np.diagflat([-0.5 * s for i in range(nx - 1)], 1)
    #
    # B1 = np.diagflat([0.5 * s for i in range(nx - 1)], -1) + \
    #      np.diagflat([1 - 0.5 * s] + [1 - s for i in range(nx - 2)] + [1 - 0.5 * s]) + \
    #      np.diagflat([0.5 * s for i in range(nx - 1)], 1)
    #
    # print(A)
    # while any(map(lambda q: np.fabs(q) > delta, (np.array(Vn) - np.array(V)))):  # time is going from second time step to last
    #     Vn = V.copy()
    #     if any(map(lambda q: np.isnan(q), Vn)):
    #         raise ValueError('Решение не найдено')
    #     B = np.dot(Vn, B1)
    #     V = solve_eq(A, B)
    #     if n % ntout == 0:
    #         Vout.append(V.copy())  # numpy arrays are mutable, so we need to write out a copy of V, not V itself
    #         print(n)
    #     n += 1
    return Cout, C


# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')

# i = 0
#
# def init():
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     return ln
#
#
#
# def update(frame, h):
#     global i
#     xdata.append(frame)
#     ydata.append(h[i])
#     ln.set_data(xdata, ydata)
#     i += 1
#     return ln

#Выводит ntout графиков
def find_time_solution(dx, nx, dt, D, V, type_GU, nt=1000000, ntout=10):
    Vout = []  # list for storing V arrays at certain time steps
    s = D * dt / dx ** 2  # diffusion number
    if type_GU == 1:
        V0 = V[0]  # boundary condition on left side
        V1 = V[-1]  # boundary condition on right side
        A = np.diagflat([-0.5 * s for i in range(nx - 3)], -1) + \
            np.diagflat([1 + s] + [1. + s for i in range(nx - 4)] + [1 + s]) + \
            np.diagflat([-0.5 * s for i in range(nx - 3)], 1)

        B1 = np.diagflat([0.5 * s for i in range(nx - 3)], -1) + \
             np.diagflat([1 - s] + [1 - s for i in range(nx - 4)] + [1 - s]) + \
             np.diagflat([0.5 * s for i in range(nx - 3)], 1)

        for n in range(1,nt): # time is going from second time step to last
            Vn = V
            B = np.dot(Vn[1:-1],B1)
            B[0] = B[0]+0.5*s*(V0+V0)
            B[-1] = B[-1]+0.5*s*(V1+V1)
            V[1:-1] = solve_eq(A, B)
            if n % int(nt/float(ntout)) == 0 or n==nt-1:
                Vout.append(V.copy()) # numpy arrays are mutable, so we need to write out a copy of V, not V itself
                print(n)

    elif type_GU == 2:
        A = np.diagflat([-0.5 * s for i in range(nx - 1)], -1) + \
            np.diagflat([1 + 0.5 * s] + [1. + s for i in range(nx - 2)] + [1 + 0.5 * s]) + \
            np.diagflat([-0.5 * s for i in range(nx - 1)], 1)

        B1 = np.diagflat([0.5 * s for i in range(nx - 1)], -1) + \
             np.diagflat([1 - 0.5 * s] + [1 - s for i in range(nx - 2)] + [1 - 0.5 * s]) + \
             np.diagflat([0.5 * s for i in range(nx - 1)], 1)

        for n in range(1,nt): # time is going from second time step to last
            Vn = V
            B = np.dot(Vn,B1)
            V = solve_eq(A, B)
            if n % int(nt/float(ntout)) == 0 or n==nt-1:
                Vout.append(V.copy()) # numpy arrays are mutable, so we need to write out a copy of V, not V itself
                print(n)
    return Vout

# Main блок
D, x_max, dx, dt, GU_left, GU_right, nt, ntout, nout, k, a1, b1, c1, a2, b2, c2 = read_const_file('Константы 1.txt')
x = np.arange(0, x_max + dx, dx)
nx = len(x)
# C_in =  np.array([1. for i in range(0, int(nx / 2))] + [0. for j in range(int(nx / 2), nx)]) # initial condition
C_in = np.array([1. for i in range(0, nx)])
C_in[-1] = 0.
if k == 1:
    C_out, C_last = find_equilibrium_solution(dx,nx, dt, D, C_in.copy(), GU_left, GU_right, a1, b1, c1, a2, b2, c2, nout)
    C_out.append(C_last)
elif k == 2:
    C_out = find_time_solution(dx, nx, dt, D, C_in.copy(), GU_left, GU_right, nt, ntout)
print(C_out[-1])

#вывод графиков
plt.plot(x , C_out[-1], linestyle='-.', color='red')
plt.xlabel('Глубина образца',fontsize=12)
plt.ylabel('Концентрация',fontsize=12)
plt.title('Конечное распределение',fontsize=14)
plt.show()
fig, ax = plt.subplots()
for C in C_out:
    if list(C) == list(C_out[-1]):
        ax.plot(x , C_out[-1], linestyle='-.', color='red',label='Конечное распределение')
    else:
        ax.plot(x, C, 'k')
ax.plot(x, C_in, linestyle='-', color='green', label='Начальное распределение')
plt.legend(loc='upper left')
plt.xlabel('Глубина образца',fontsize=12)
plt.ylabel('Концентрация',fontsize=12)
plt.title('Crank-Nicolson scheme',fontsize=14)
plt.show()




# fig, ax = plt.subplots()
# x = np.linspace(0, 1, b.shape[0])
# for iteration, solution in res_list:
#     if iteration == res_list[-1][0]:
#         ax.plot(x, solution, label='Конечное распределение', ls='-.')
#     else:
#         ax.plot(x, solution)
#
# ax.plot(x, b, label='Начальное распределение', ls='--')
# plt.legend(loc='upper left')
# plt.ylabel("C")
# plt.xlabel("x")
# plt.savefig('1.png')
# plt.show()
#
# print('Решение найдено:', '\n', y)

# h = mat_ic.dot(b)
# x, y = solve_two_first_eq(mat, h)[x], solve_two_first_eq(mat,  h)[y]
# print(x.args[0], x.args[1].as_coeff_mul()[1][0]*x.args[1].as_coeff_mul()[0], y.args[0], y.args[1].as_coeff_mul()[1][0]*y.args[1].as_coeff_mul()[0], x, y)
# solution_two_first_eq = solve_two_first_eq(mat, b)
# print(solution_two_first_eq)
