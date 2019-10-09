from matplotlib import pyplot as plt


def plot_result(names, colors):
    plt.figure('result')
    plt.title('result')

    for name, color in zip(names, colors):
        x = []
        y = []

        file_path = 'results/{}/top_result.txt'.format(name)
        for line in open(file_path).readlines():
            topk, acc = line.strip().split('\t')
            x.append(int(topk))
            y.append(float(acc))

        plt.plot(x, y, color=color, label=name)

    plt.legend()
    plt.xlabel('retrieved images')
    plt.ylabel('retrieval accuracy')


def plot_log(name):
    plt.figure(name)
    plt.title(name)

    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    file_path = 'logs/{}.txt'.format(name)
    for line in open(file_path).readlines():
        if 'val loss' in line:
            iter, val_loss = line.strip().split(',')
            iter = int(iter.split(' ')[-1])
            val_loss = float(val_loss.split(' ')[-1])
            # test_loss = float(test_loss.split(' ')[-1])
            val_x.append(iter)
            val_y.append(val_loss)
            # test_x.append(iter)
            # test_y.append(test_loss)
        elif 'lr' in line:
            iter, _, train_loss = line.strip().split(',')[:3]
            iter = int(iter.split(' ')[-1])
            train_loss = float(train_loss.split(' ')[-1])
            train_x.append(iter)
            train_y.append(train_loss)

    plt.plot(train_x, train_y, color='green', label='train')
    plt.plot(val_x, val_y, color='blue', label='val')
    # plt.plot(test_x, test_y, color='red', label='test')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')


if __name__ == '__main__':
    plot_result(names=['random_128', 'resnet50_128_loss.batch_hard_loss_euclidean_0.2', 'resnet50_128_loss.margin_based_loss_euclidean_0.2',
                       'resnet50_128_loss.margin_sample_mining_loss_euclidean_0.2',
                       # 'resnet50_blur_128_loss.batch_hard_loss_euclidean_0.2',
                       'resnet50_512_loss.batch_hard_loss_euclidean_0.2',
                       # 'resnet50_512_loss.batch_hard_loss_euclidean_soft',
                       'resnet50_128_loss.ms_loss_cosine_1.0'],
                colors=['black', 'red', 'green',
                        # 'blue',
                        # 'pink',
                        'yellow', 'orange', 'gray'])

    plt.show()
