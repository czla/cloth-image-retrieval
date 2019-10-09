import os


def read_data(data_path):
    data = list(open(data_path).readlines())
    data = data[2:]

    return data


def split_data(line):
    line_split = line.strip().split(' ')
    line_split = list(filter(lambda line: line, line_split))

    return line_split


if __name__ == '__main__':
    image_type = {}
    for line in read_data('Anno/list_bbox_consumer2shop.txt'):
        line_split = split_data(line)
        image_name, clothes_type, source_type, x_1, y_1, x_2, y_2 = line_split
        image_type[image_name] = clothes_type

    data = {'train': {}, 'val': {}, 'test': {}}
    for line in read_data('Eval/list_eval_partition.txt'):
        line_split = split_data(line)
        pos, anchor, item_id, evaluation_status = line_split
        if image_type[pos] == image_type[anchor] == '1':
            if anchor not in data[evaluation_status]:
                data[evaluation_status][anchor] = set([])
            data[evaluation_status][anchor].add(pos)

    root_path = '/DATA2/data/yjgu/deepFashion/C2SCR'
    for evaluation_status in data.keys():
        with open('%s.txt' % evaluation_status, 'w') as f:
            for anchor in data[evaluation_status].keys():
                all_path = [anchor] + list(data[evaluation_status][anchor])
                if len(all_path) >= 3:
                    all_path = list(map(lambda line: os.path.join(root_path,line), all_path))
                    all_path = '\t'.join(all_path)
                    f.write('%s\n' % all_path)

    # with open('testConsumer.txt', 'w') as f1, open('testShop.txt', 'w') as f2:
    #     n = 0
    #     for anchor in data['test'].keys():
    #         anchor_path = os.path.join(root_path, anchor)
    #         f2.write('%s\n' % anchor_path)
    #         for pos in data['test'][anchor]:
    #             pos_path = os.path.join(root_path, pos)
    #             f1.write('%s\t%d\n' % (pos_path, n))
    #         n += 1
