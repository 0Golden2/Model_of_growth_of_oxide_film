import configparser
from functions import exp, gauss, polinom
import numpy as np


def get_config(path):
    """
    Returns the config object
    """
    # if not os.path.exists(path):
    #     create_config(path)

    config = configparser.ConfigParser()
    config.read(path)
    return config


class param:
    def __init__(self,  path):
        config = get_config(path)
        dx = []
        dt = []
        D = []
        l0 = []
        A_l = []
        B_l = []
        C_l = []
        A_r = []
        B_r = []
        C_r = []
        x1 = []
        x2 = []
        C_in1 = []
        C_in2 = []
        types_func = {}
        parameters = {}
        norm = {}
        path = {}
        # num_of_points = {}
        num_of_iter = []
        steps_of_print = []
        num_of_graphs = []
        num_of_eq = config.getint('num_of_eq', 'n')
        for i in range(1, num_of_eq + 1):
            D.append(config.getfloat(f'constants_of_{i}_eq', 'D'))
            l0.append(config.getfloat(f'constants_of_{i}_eq', 'l0'))
            dx.append(config.getfloat(f'constants_of_{i}_eq', 'dx'))
            dt.append(config.getfloat(f'constants_of_{i}_eq', 'dt'))

            A_l.append(config.getfloat(f'l_boundary_of_{i}_eq', 'A'))
            B_l.append(config.getfloat(f'l_boundary_of_{i}_eq', 'B'))
            C_l.append(config.getfloat(f'l_boundary_of_{i}_eq', 'C'))

            A_r.append(config.getfloat(f'r_boundary_of_{i}_eq', 'A'))
            B_r.append(config.getfloat(f'r_boundary_of_{i}_eq', 'B'))
            C_r.append(config.getfloat(f'r_boundary_of_{i}_eq', 'C'))

            types_func[i] = config.get(f'C_init_of_{i}_eq', 'type')
            parameters[i] = [float(j.strip()) for j in config.get(f'C_init_of_{i}_eq', 'parameters').split(',')]
            norm[i] = config.getboolean(f'C_init_of_{i}_eq', 'norm')
            path[i] = config.get(f'C_init_of_{i}_eq', 'path')

            if i == 1:
                x1 = np.arange(0, l0[0] + dx[0], dx[0])
                if types_func[i] == 'exp':
                    C_in1 = exp(x1, norm[i])
                elif types_func[i] == 'polinom':
                    C_in1 = polinom(x1, parameters[i], norm[i])
                elif types_func[i] == 'gauss':
                    C_in1 = gauss(x1, parameters[i], norm[i])
                elif types_func[i] == 'file':
                    y = np.loadtxt(path[i], usecols=0)
                    if any(map(lambda q: q > 1 or q < 0, y)):
                        C_in1 = y / max(y)
                    else:
                        if norm[i] and max(y) != 0:
                            C_in1 = y / max(y)
                        else:
                            C_in1 = y
            if i == 2:
                x2 = np.arange(0, l0[1] + dx[1], dx[1])
                if types_func[i] == 'exp':
                    C_in2 = exp(x2, norm[i])
                elif types_func[i] == 'polinom':
                    C_in2 = polinom(x2, parameters[i], norm[i])
                elif types_func[i] == 'gauss':
                    C_in2 = gauss(x2, parameters[i], norm[i])
                elif types_func[i] == 'file':
                    y = np.loadtxt(path[i], usecols=0)
                    if any(map(lambda q: q > 1 or q < 0, y)):
                        C_in2 = y / max(y)
                    else:
                        if norm[i] and max(y) != 0:
                            C_in1 = y / max(y)
                        else:
                            C_in2 = y
            # num_of_points[i] = [int(j.strip()) for j in config.get(f'C_init_of_{i}_eq', 'num_of_points').split(',')]

            # eq_c = dict(zip(eq_val, eq_break_points))
            # C_in = []
            # bp_prev = 0
            # for val, bp in eq_c.items():
            #     C_in += [val for i in range(bp_prev, bp)]
            #     bp_prev = bp
            # if i == 1:
            #     C_in1 = C_in.copy()
            # elif i == 2:
            #     C_in2 = C_in.copy()

            num_of_iter.append(config.getint(f'constants_for_printing_graphs_of_{i}_eq', 'num_of_iter'))
            steps_of_print.append(config.getint(f'constants_for_printing_graphs_of_{i}_eq', 'steps_of_print'))
            num_of_graphs.append(config.getint(f'constants_for_printing_graphs_of_{i}_eq', 'num_of_graphs'))
        model_oxide = config.getboolean('model_oxide', 'model_oxide')
        lamb = config.getfloat('model_oxide', 'lambda')
        self.dx = dx
        self.dt = dt
        self.D = D
        self.l0 = l0
        self.num_of_eq = num_of_eq
        self.A_l = A_l
        self.B_l = B_l
        self.C_l = C_l
        self.A_r = A_r
        self.B_r = B_r
        self.C_r = C_r
        # self.types_func = types_func
        self.C_in1 = C_in1
        self.C_in2 = C_in2
        self.x1 = x1
        self.x2 = x2
        self.num_of_iter = num_of_iter
        self.steps_of_print = steps_of_print
        self.num_of_graphs = num_of_graphs
        self.model_oxide = model_oxide
        self.lamb = lamb
        # eq_d_break_points = [int(i.strip()) for i in config.get('D_coeff', 'break_points').split(',')]
        # eq_d_equations = [(i.strip()) for i in config.get('D_coeff', 'equations').split(',')]
        # eq_d = dict(zip(eq_d_equations, eq_d_break_points))
        # self.eq_d = eq_d
        #
        # eq_g_break_points = [int(i.strip()) for i in config.get('g_x', 'break_points').split(',')]
        # eq_g_equations = [(i.strip()) for i in config.get('g_x', 'equations').split(',')]
        # eq_g = dict(zip(eq_g_equations, eq_g_break_points))
        # self.eq_g = eq_g