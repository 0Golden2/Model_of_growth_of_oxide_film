import configparser
import os


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
        C_in1 = []
        C_in2 = []
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

            eq_break_points = [int(i.strip()) for i in config.get(f'C_init_of_{i}_eq', 'break_points').split(',')]
            eq_val = [float(i.strip()) for i in config.get(f'C_init_of_{i}_eq', 'values').split(',')]
            eq_c = dict(zip(eq_val, eq_break_points))
            C_in = []
            bp_prev = 0
            for val, bp in eq_c.items():
                C_in += [val for i in range(bp_prev, bp)]
                bp_prev = bp
            if i == 1:
                C_in1 = C_in.copy()
            elif i == 2:
                C_in2 = C_in.copy()

            num_of_iter.append(config.getint(f'constants_for_printing_graphs_of_{i}_eq', 'num_of_iter'))
            steps_of_print.append(config.getint(f'constants_for_printing_graphs_of_{i}_eq', 'steps_of_print'))
            num_of_graphs.append(config.getint(f'constants_for_printing_graphs_of_{i}_eq', 'num_of_graphs'))

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
        self.C_in1 = C_in1
        self.C_in2 = C_in2
        self.num_of_iter = num_of_iter
        self.steps_of_print = steps_of_print
        self.num_of_graphs = num_of_graphs
        # eq_d_break_points = [int(i.strip()) for i in config.get('D_coeff', 'break_points').split(',')]
        # eq_d_equations = [(i.strip()) for i in config.get('D_coeff', 'equations').split(',')]
        # eq_d = dict(zip(eq_d_equations, eq_d_break_points))
        # self.eq_d = eq_d
        #
        # eq_g_break_points = [int(i.strip()) for i in config.get('g_x', 'break_points').split(',')]
        # eq_g_equations = [(i.strip()) for i in config.get('g_x', 'equations').split(',')]
        # eq_g = dict(zip(eq_g_equations, eq_g_break_points))
        # self.eq_g = eq_g


def create_config(path):
    """
    Create a config file
    """
    config = configparser.ConfigParser()
    config.add_section('num_of_eq')
    config.set('num_of_eq', 'n', '2')

    config.add_section('constants_of_1_eq')
    config.set('constants_of_1_eq', 'D', '0.1')
    config.set('constants_of_1_eq', 'l0', '1')
    config.set('constants_of_1_eq', 'dx', '0.01')
    config.set('constants_of_1_eq', 'dt', '1.0')
    config.set('constants_of_1_eq', 'D', '0.1')

    config.add_section('l_boundary_of_1_eq')
    config.set('l_boundary_of_1_eq', 'A', '0.0')
    config.set('l_boundary_of_1_eq', 'B', '1.0')
    config.set('l_boundary_of_1_eq', 'С', '1.0')

    config.add_section('r_boundary_of_1_eq')
    config.set('r_boundary_of_1_eq', 'A', '0.0')
    config.set('r_boundary_of_1_eq', 'B', '1.0')
    config.set('r_boundary_of_1_eq', 'С', '0.0')

    config.add_section('C_init_of_1_eq')
    config.set('C_init_of_1_eq', 'break_points', '100, 101')
    config.set('C_init_of_1_eq', 'values', '1.0, 0.0')

    config.add_section('constants_for_printing_graphs_of_1_eq')
    config.set('constants_for_printing_graphs_of_1_eq', 'num_of_iter', 'None')
    config.set('constants_for_printing_graphs_of_1_eq', 'steps_of_print', '10')
    config.set('constants_for_printing_graphs_of_1_eq', 'num_of_graphs', '100')

    config.add_section('constants_of_2_eq')
    config.set('constants_of_2_eq', 'D', '0.1')
    config.set('constants_of_2_eq', 'l0', '1')
    config.set('constants_of_2_eq', 'dx', '0.01')
    config.set('constants_of_2_eq', 'dt', '1.0')
    config.set('constants_of_2_eq', 'D', '0.1')

    config.add_section('l_boundary_of_2_eq')
    config.set('l_boundary_of_2_eq', 'A', '0.0')
    config.set('l_boundary_of_2_eq', 'B', '1.0')
    config.set('l_boundary_of_2_eq', 'D', '1.0')

    config.add_section('r_boundary_of_2_eq')
    config.set('r_boundary_of_2_eq', 'A', '0.0')
    config.set('r_boundary_of_2_eq', 'B', '1.0')
    config.set('r_boundary_of_2_eq', 'D', '0.0')

    config.add_section('C_init_of_2_eq')
    config.set('C_init_of_2_eq', 'break_points', '50, 101')
    config.set('C_init_of_2_eq', 'values', '1.0, 0.0')

    config.add_section('constants_for_printing_graphs_of_2_eq')
    config.set('constants_for_printing_graphs_of_2_eq', 'num_of_iter', 'None')
    config.set('constants_for_printing_graphs_of_2_eq', 'steps_of_print', '10')
    config.set('constants_for_printing_graphs_of_2_eq', 'num_of_graphs', '100')
    # config.add_section('D_coeff')
    # config.set('D_coeff', 'break_points', '1000')
    # config.set('D_coeff', 'equations', '0.5')
    #
    # config.add_section('g_x')
    # config.set('g_x', 'break_points', '1000')
    # config.set('g_x', 'equations', '0.0')
    with open(path, 'w') as config_file:
        config.write(config_file)


def get_config(path):
    """
    Returns the config object
    """
    if not os.path.exists(path):
        create_config(path)

    config = configparser.ConfigParser()
    config.read(path)
    return config
