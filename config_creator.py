import configparser


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
    config.set('constants_of_1_eq', 'dt', '0.001')

    config.add_section('l_boundary_of_1_eq')
    config.set('l_boundary_of_1_eq', 'A', '0.0')
    config.set('l_boundary_of_1_eq', 'B', '1.0')
    config.set('l_boundary_of_1_eq', 'C', '-1.0')

    config.add_section('r_boundary_of_1_eq')
    config.set('r_boundary_of_1_eq', 'A', '0.0')
    config.set('r_boundary_of_1_eq', 'B', '1.0')
    config.set('r_boundary_of_1_eq', 'C', '0.0')

    config.add_section('C_init_of_1_eq')
    config.set('C_init_of_1_eq', 'type', 'exp')
    config.set('C_init_of_1_eq', 'parameters', '-1')
    config.set('C_init_of_1_eq', 'path', '')

    config.add_section('constants_for_printing_graphs_of_1_eq')
    config.set('constants_for_printing_graphs_of_1_eq', 'num_of_iter', '10000')
    config.set('constants_for_printing_graphs_of_1_eq', 'steps_of_print', '10')
    config.set('constants_for_printing_graphs_of_1_eq', 'num_of_graphs', '100')

    config.add_section('constants_of_2_eq')
    config.set('constants_of_2_eq', 'D', '0.1')
    config.set('constants_of_2_eq', 'l0', '1')
    config.set('constants_of_2_eq', 'dx', '0.01')
    config.set('constants_of_2_eq', 'dt', '0.001')

    config.add_section('l_boundary_of_2_eq')
    config.set('l_boundary_of_2_eq', 'A', '0.0')
    config.set('l_boundary_of_2_eq', 'B', '1.0')
    config.set('l_boundary_of_2_eq', 'C', '-1.0')

    config.add_section('r_boundary_of_2_eq')
    config.set('r_boundary_of_2_eq', 'A', '1.0')
    config.set('r_boundary_of_2_eq', 'B', '0.0')
    config.set('r_boundary_of_2_eq', 'C', '0.0')

    config.add_section('C_init_of_2_eq')
    config.set('C_init_of_2_eq', 'type', 'polinom')
    config.set('C_init_of_2_eq', 'parameters', '1, 1, 1')
    config.set('C_init_of_2_eq', 'path', '')

    config.add_section('constants_for_printing_graphs_of_2_eq')
    config.set('constants_for_printing_graphs_of_2_eq', 'num_of_iter', '10000')
    config.set('constants_for_printing_graphs_of_2_eq', 'steps_of_print', '10')
    config.set('constants_for_printing_graphs_of_2_eq', 'num_of_graphs', '100')

    config.add_section('model_oxide')
    config.set('model_oxide', 'model_oxide', 'False')
    # config.add_section('D_coeff')
    # config.set('D_coeff', 'break_points', '1000')
    # config.set('D_coeff', 'equations', '0.5')
    #
    # config.add_section('g_x')
    # config.set('g_x', 'break_points', '1000')
    # config.set('g_x', 'equations', '0.0')
    with open(path, 'w') as config_file:
        config.write(config_file)

