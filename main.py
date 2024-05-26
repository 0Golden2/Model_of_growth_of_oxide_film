from solver import solver
from data_saver import print_plot, write_data
from config_reader import read_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

path = input('Input path of init file. If you want to exit, type \'exit\'\n')
params = read_config(path)
if params is not None:
    # Unpacking initial parameters, constants and distributions
    num_eq = params.num_of_eq
    cnst = [params.D, params.l0, params.dx, params.dt, params.A_l,
            params.B_l, params.C_l, params.A_r, params.B_r, params.C_r]
    ntout = [params.num_of_iter, params.num_of_graphs, params.steps_of_print]
    model_oxide = params.model_oxide
    lamb = params.lamb

    x1 = params.x1
    x2 = params.x2
    C_in1 = params.C_in1
    C_in2 = params.C_in2

    # Find solution with solver
    C1_out, C1_last, C2_out, C2_last, x1_out, x2_out, arr_n_l, arr_n_r, arr_dL_l, arr_dL_r, flow_ox, flow_fe, dif = \
        solver(cnst, ntout, num_eq, C_in1.copy(), x1.copy(), C_in2.copy(), x2.copy(), model_oxide, lamb)
    # print(x1)
    print(arr_n_r, arr_n_l)
    C1_out.append(C1_last)
    C2_out.append(C2_last)
    # print(len(arr_n_l), len(arr_n_r), len(C1_out), len(C2_out))
    # print([len(row) for row in C2_out])
    max_len1 = max(len(row) for row in C1_out)
    # print(max_len1)
    for i in range(len(C1_out)):
        ext_arr = [0] * (max(arr_n_l) - arr_n_l[i])
        ext_arr.extend(C1_out[i])
        C1_out[i] = ext_arr
        C1_out[i] = np.append(C1_out[i], [0] * (max(arr_n_r) - arr_n_r[i]))
        # print(len(C1_out[i]))
            # print(C1_out[i])
            # break
    C1_out = list(np.clip(np.array(C1_out), 0, 1))
    # arr = [1, 2, 3, 4, 5, 6, 7, 8]
    max_len2 = max(len(row) for row in C2_out)
    for i in range(len(C2_out)):
        ext_arr = [0] * (max(arr_n_l) - arr_n_l[i])
        ext_arr.extend(C2_out[i])
        C2_out[i] = ext_arr
        C2_out[i] = np.append(C2_out[i], [0] * (max(arr_n_r) - arr_n_r[i]))
            # print(len(C2_out[i]))
    # print(max([len(row) for row in C1_out]), min([len(row) for row in C1_out]))
    # print(max([len(row) for row in C2_out]), min([len(row) for row in C2_out]))
    # for i in range(len(C2_out)):
    #     if len(C2_out[i]) != 185:
    #         print(len(C2_out[i]))
    C2_out = list(np.clip(np.array(C2_out), 0, 1))

    arr_dL_l = np.array(arr_dL_l) / 100
    arr_dL_r = np.array(arr_dL_r) / 100
    arr_dL = np.array(arr_dL_l) - np.array(arr_dL_r)
    # Writing data into txt file(s)
    write_data(1, C1_out)
    if num_eq == 2:
        write_data(2, C2_out)

    t = np.arange(1, ntout[0][0])

    fig0, ax0 = plt.subplots()

    ax0.plot(t, arr_dL_l, label='Толщина левой пленки')
    ax0.plot(t, arr_dL_r, label='Толщина правой пленки')
    ax0.plot(t, arr_dL, label='Суммарная толщина пленки')
    ax0.set_xlabel('t')
    ax0.set_ylabel('L')
    plt.title(f"""График роста пленки от времени с параметрами:
                D_fet(T) / D_ot(T) = {dif:.6f}, dt = {cnst[3][1]}, кол-во итераций = {ntout[0][0]}""", fontsize=9)
    plt.legend(loc='lower right')
    plt.savefig(
        r'C:\Users\Skl\Documents\Proga\4 курс\Отсчеты Бородин\Работа программы с разными параметрами\Рост пленки 1.jpeg')

    arr_dL_log = np.log10(arr_dL[1:])
    arr_dL_r_log = np.log10(np.abs(arr_dL_r[1:]))
    arr_dL_l_log = np.log10(arr_dL_l[1:])

    t_log = np.log10(t[1:])
    # Output of graphs
    # plt.plot(x2_out, C2_out[-1])
    # plt.show()
    # plt.savefig(
    #     r'C:\Users\Skl\Documents\Proga\4 курс\Отсчеты Бородин\Работа программы с разными параметрами\Фин распределение.jpeg')
    print_plot(x1, C_in1, dif, cnst[3][1], ntout[0][0], x1_out, C1_out, 1)
    if num_eq == 2:
        print_plot(x2, C_in2, dif, cnst[3][1], ntout[0][0], x2_out, C2_out, 2)
    arr1 = [0.5 * x for x in t_log]
    fig, ax = plt.subplots()

    ax.plot(t_log, arr_dL_l_log, label='Толщина левой пленки')
    ax.plot(t_log, arr_dL_r_log, label='Толщина правой пленки')
    ax.plot(t_log, arr_dL_log, label='Суммарная толщина пленки')
    ax.plot(t_log, arr1, label='y = 0.5x')
    ax.set_xlabel('log(t)')
    ax.set_ylabel('log(L)')
    plt.title(f"""График роста пленки от времени с параметрами:
            D_fet(T) / D_ot(T) = {dif:.6f}, dt = {cnst[3][1]}, кол-во итераций = {ntout[0][0]}""", fontsize=9)
    plt.legend(loc='lower right')
    plt.savefig(
        r'C:\Users\Skl\Documents\Proga\4 курс\Отсчеты Бородин\Работа программы с разными параметрами\Рост пленки в лог масштабе1.jpeg')
    # plt.show()
    dL_grad = np.gradient(arr_dL_log, t_log)
    arr2 = [0.5 for x in t_log]

    fig1, ax1 = plt.subplots()
    ax1.plot(t_log, dL_grad, label='График производной')
    ax1.plot(t_log, arr2, label='y = 0.5')

    ax1.set_xlabel('log(t)')
    ax1.set_ylabel('d(log(L))/d(log(t))')

    plt.legend(loc='upper right')

    # dL_grad = np.gradient(arr1, t_log)
    # plt.plot(t_log, arr1)
    plt.title(f"""Производная графика роста пленки от времени с параметрами:
                D_fet(T) / D_ot(T) = {dif:.6f}, dt = {cnst[3][1]}, кол-во итераций = {ntout[0][0]}""", fontsize=9)
    plt.savefig(
        r'C:\Users\Skl\Documents\Proga\4 курс\Отсчеты Бородин\Работа программы с разными параметрами\Призв Рост пленки в лог масштабе1.jpeg')
    # plt.show()
    print(dL_grad)
    # print_plot(arr_dL_l, flow_ox, x_label='Толщина пленки на левой границе', y_label='flow_ox')
    # print_plot(t, flow_fe, x_label='Time', y_label='flow_fe')
    print(arr_dL_r[-1], arr_dL_l[-1])
    # print_plot(t, arr_dL_l, x_label='Time')
    # print_plot(t, arr_dL_r, x_label='Time')
