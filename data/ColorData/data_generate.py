import random

if __name__ == '__main__':
    train_ratio = 0.8
    val_ratio = 0.1

    data = list(open('data.txt').readlines())
    random.shuffle(data)

    train_num = int(train_ratio*len(data))
    val_num =  int(val_ratio*len(data))

    data_split = {}
    data_split['train'] = data[:train_num]
    data_split['val'] = data[train_num:train_num+val_num]
    data_split['test'] = data[train_num+val_num:]

    for key in data_split.keys():
        with open('%s.txt' % key, 'w') as f:
            for line in data_split[key]:
                f.write(line)