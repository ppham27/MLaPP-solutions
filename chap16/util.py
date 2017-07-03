from scipy import io
import pandas as pd

TARGET_COLUMN = 'y1'

def read_sarcos_data(dataset):
    file_name = dataset + '.mat'
    raw_data = io.loadmat(file_name)
    column_names = list()
    ## first 21 variables are input
    for i in range(1, 22):
        column_names.append('x' + str(i))
    ## next 7 variables are output
    for i in range(1, 8):
        column_names.append('y' + str(i))
    return pd.DataFrame(raw_data[dataset], columns=column_names)

def read_all_sarcos_data():
    return read_sarcos_data('sarcos_inv'), read_sarcos_data('sarcos_inv_test')

if __name__ == "__main__":
    print(read_sarcos_data('sarcos_inv').head())
    print(read_sarcos_data('sarcos_inv').tail())
    print(read_sarcos_data('sarcos_inv').info())
    print(read_sarcos_data('sarcos_inv_test').head())
    print(read_sarcos_data('sarcos_inv_test').tail())
    print(read_sarcos_data('sarcos_inv_test').info())
