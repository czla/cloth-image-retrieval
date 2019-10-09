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
    data = {'train': {}, 'query': {}, 'gallery': {}, 'val': {}}
    for line in read_data('Eval/list_eval_partition.txt'):
        line_split = split_data(line)
        image_name, item_id, evaluation_status = line_split
        if item_id not in data[evaluation_status]:
            data[evaluation_status][item_id] = set([])
        data[evaluation_status][item_id].add(image_name)

        if evaluation_status != 'train':
            if item_id not in data['val']:
                data['val'][item_id] = set([])
            data['val'][item_id].add(image_name)

    with open('train.txt', 'w') as f:
        for item_id in data['train']:
            line_str = '\t'.join(data['train'][item_id])
            f.write('%s\n' % line_str)

    with open('val.txt', 'w') as f:
        for item_id in data['val']:
            line_str = '\t'.join(data['val'][item_id])
            f.write('%s\n' % line_str)

    labels = {}
    id = 0
    for item_id in data['gallery']:
        labels[item_id] = id
        id += 1

    with open('query.txt', 'w') as f, open('query_label.txt', 'w') as g:
        for item_id in data['query']:
            for image_name in data['query'][item_id]:
                f.write('%s\n' % image_name)
                g.write('%d\n' % labels[item_id])

    with open('gallery.txt', 'w') as f, open('gallery_label.txt', 'w') as g:
        for item_id in data['gallery']:
            for image_name in data['gallery'][item_id]:
                f.write('%s\n' % image_name)
                g.write('%d\n' % labels[item_id])
