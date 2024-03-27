import os
from config_parser import param
from config_creator import create_config


def read_config(path):
    params = None
    while True:
        # Parameters get from text file, that locates in the same directory, as this .py file
        if os.path.isfile(path):
            try:
                params = param(path)
            except ValueError as e:
                create_config('default.ini')
                params = param('default.ini')
                print(f'Wrong values in init file:{e}. Data will be taken from the default file')
            print('Please wait the calculation is underway')
            break
        else:
            if path == 'exit':
                break
            path = '________'
            # a loop that ends when the file is written correctly
            while not (os.path.isfile(path)):
                path = input('Input path of init file. If you want to exit, type \'exit\'\n')
                if path == 'exit':
                    break
                if os.path.isfile(path):
                    break
                try:
                    params = param(path)
                except Exception:
                    print('\nWrong name of file, it doesn\'t exist in diractory. Try again.')
    return params
