from solver import solver
from data_saver import print_plot, write_data
from config_reader import read_config


path = input('Input path of init file. If you want to exit, type \'exit\'\n')
params = read_config(path)
if params is not None:
    # Unpacking initial parameters, constants and distributions
    num_eq = params.num_of_eq
    cnst = [params.D, params.l0, params.dx, params.dt, params.A_l,
            params.B_l, params.C_l, params.A_r, params.B_r, params.C_r]
    ntout = [params.num_of_iter, params.num_of_graphs, params.steps_of_print]
    model_oxide = params.model_oxide
    x1 = params.x1
    x2 = params.x2
    C_in1 = params.C_in1
    C_in2 = params.C_in2

    # Find solution with solver
    C1_out, C1_last, C2_out, C2_last = solver(cnst, ntout, num_eq,  C_in1.copy(), x1.copy(), C_in2.copy(), x2.copy(),
                                              model_oxide)
    C1_out.append(C1_last)
    C2_out.append(C2_last)

    # Writing data into txt file(s)
    write_data(1, C1_out)
    if num_eq == 2:
        write_data(2, C2_out)

    # Output of graphs
    print_plot(x1, C1_out, C_in1, 1)
    if num_eq == 2:
        print_plot(x2, C2_out, C_in2, 2)
