import numpy as np
import itertools
from scipy.spatial.distance import cdist


def get_permutations(grid, class_num):
    n = grid[0] * grid[1]
    P_hat = np.array(list(itertools.permutations(list(range(n)), n)))

    for i in range(class_num):
        if i == 0:
            j = 0
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1,-1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        j = D.argmax()

    return P


def random_jigsaw(image, grid, permutations):
    h, w, _ = image.shape
    gh, gw = grid
    grid_h, grid_w = h // gh, w // gw

    num_class = len(permutations)
    order = np.random.randint(0, num_class)
    permutation = permutations[order]

    new_image = np.zeros_like(image)
    for i in range(gh):
        for j in range(gw):
            pos = permutation[i*gw+j]
            org_i = int(pos / gw)
            org_j = pos - org_i * gw
            new_image[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_h] = image[org_i*grid_h:(org_i+1)*grid_h,org_j*grid_w:(org_j+1)*grid_w]

    order_out = [0 for _ in range(num_class)]
    order_out[order] = 1

    return new_image, order_out


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt

    grid = (3,3)
    class_num = 100
    permutations = get_permutations(grid, class_num)

    image = cv2.imread('/home/hxcai/Pictures/test/1563433082069_1_13_1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (192,192))

    new_image, order = random_jigsaw(image, grid, permutations)

    plt.figure(1)
    plt.imshow(image.astype(np.int32))
    plt.figure(2)
    plt.imshow(new_image.astype(np.int32))
    plt.show()

    print(order)