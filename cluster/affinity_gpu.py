from time import time

import torch
from torch.nn.functional import pdist, pairwise_distance
import numpy as np
from scipy.spatial.distance import squareform, pdist as scipy_dist

from utils import load_samples


def calculate_affinity_gpu(data):
    data = torch.tensor(data).float().cuda()  # type: torch.Tensor

    n, d = data.shape

    # t = time()
    distances = pdist(data)
    # print('pdist', time() - t)

    distances_mat = torch.zeros((n, n)).float().cuda()

    # t = time()
    cur_idx = 0
    for i in range(n):
        distances_mat[i, i + 1:] = distances[cur_idx: cur_idx + (n - i - 1)]
        cur_idx += (n - i - 1)
    # print(cur_idx)
    # print('dist_mat', time() - t)
    # t = time()
    distances_mat = distances_mat + distances_mat.T

    # print(distances_mat.shape)

    # print('dist.T', time() - t)
    # t = time()
    rank = torch.zeros((n, n)).float().cuda()
    for i in range(n):
        rank[i] = torch.argsort(torch.argsort(distances_mat[i]))
    # print('rank', time() - t)
    #
    affinity = rank * rank.T

    return affinity.cpu().numpy()


if __name__ == '__main__':
    tt = time()
    # aff_0 = load_affinity('face', 'Client-0', 'local')
    # data, _ = load_samples('face', 'Client-0', 'local')
    data = np.random.randint(0, 256, (6000, 3 * 32 * 32))
    # aff_0 = calculate_affinity_cpu(data)
    # aff_1 = calculate_affinity_gpu(data)
    # print(aff_0)
    # print(aff_1)
    # print(aff_0 - aff_1)
    print(data.min(), data.max())
    # print((aff_0 - aff_2).sum())
    print(time() - tt)
