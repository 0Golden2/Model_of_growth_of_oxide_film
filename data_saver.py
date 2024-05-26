import matplotlib.pyplot as plt
import numpy as np


def print_plot(x, C_in, D, dt, n, x_out=None, C_out=None, i=1, x_label='Толщина пленки', y_label='Концентрация'):
    # plt.plot(x, C_out[-1], linestyle='-.', color='red')
    # plt.xlabel('Глубина образца', fontsize=12)
    # plt.ylabel('Концентрация', fontsize=12)
    # plt.title('Конечное распределение', fontsize=14)
    # plt.show()
    fig, ax = plt.subplots()
    if x_out is not None and C_out is not None:
        for C in C_out:
            if list(C) == list(C_out[-1]):
                ax.plot(x_out, C_out[-1], linestyle='-.', color='red', label='Конечное')
            else:
                ax.plot(x_out, C, 'k')
    ax.plot(x, C_in, linestyle='-', color='green', label='Начальное')
    plt.legend(loc='upper left')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    if i == 1:
        plt.title(f"""Распределение кислорода с параметрами: 
                D_fet(T) / D_ot(T) = {D:.6f}, dt = {dt}, кол-во итераций = {n}""", fontsize=9)
        plt.savefig(r"C:\Users\Skl\Documents\Proga\4 курс\Отсчеты Бородин\Работа программы с разными параметрами\Distribution of oxygen 1.jpeg")
        # plt.show()
    elif i == 2:
        plt.title(f"""Распределение атомов железа с параметрами: 
                D_fet(T) / D_ot(T) = {D:.6f}, dt = {dt}, кол-во итераций = {n}""", fontsize=9)
        plt.savefig(r'C:\Users\Skl\Documents\Proga\4 курс\Отсчеты Бородин\Работа программы с разными параметрами\Distribution of Fe 1.jpeg')
        # plt.show()


def write_data(k, C_out):
    with open(f"Results of the equation {k}.txt", "w+", encoding='utf-8') as f:
        num_steps = len(C_out)
        # C_out = np.array(C_out)
        C_out_clipped = np.clip(C_out, 0, 1)
        # Создаем список заголовков для каждого столбца
        headers = ['time_step_{:<5}'.format(i) for i in range(1, len(C_out_clipped) + 1)]
        for i in range(num_steps):
            if i % 10 == 0:
                np.savetxt(f, C_out_clipped[:10].T, fmt='%10.13f', delimiter=' ', header=' '.join(headers[:10]), comments='')
                C_out_clipped = C_out_clipped[10:]
                headers = headers[10:]
    print(f"The results of the equation {k} have been successfully recorded")
