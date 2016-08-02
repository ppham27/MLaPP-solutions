from scipy import io
import pandas as pd

def read_spam_data():
    raw_data = io.loadmat('spamData.mat')
    with open('spambase.names') as f:
        current_line = f.readline().strip()
        ## get response line
        while current_line == '' or current_line[0] == '|':
            current_line = f.readline().strip()
        split_response = current_line.split('|')
        split_response = list(map(lambda x : x.split(), split_response))
        for i in range(len(split_response)):
            split_response[i] = list(map(lambda x : x.strip(',.'), split_response[i]))
        names = dict()
        names['response'] = dict()
        for i in range(len(split_response[0])):
            names['response'][split_response[0][i]] = split_response[1][i]
        names['X_columns'] = list()
        for line in f:
            l = line.strip()
            if l != '' and l[0] != '|':
                names['X_columns'].append(l.split(':')[0])
    train_data = pd.DataFrame(raw_data['Xtrain'], columns=names['X_columns'])
    train_data.insert(0, column=names['response']['1'], value=raw_data['ytrain'])
    test_data = pd.DataFrame(raw_data['Xtest'], columns=names['X_columns'])
    test_data.insert(0, column=names['response']['1'], value=raw_data['ytest'])
    return train_data, test_data
