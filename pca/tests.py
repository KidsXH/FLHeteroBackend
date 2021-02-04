import json
import numpy as np
from pca import load_samples, pca_show_figure, grouping, show_figure, get_pca_result
from pca.cpca import CPCA
from utils import chebyshev_distance, grid, feature_standardize

radius = 6
step_size = 1

result = get_pca_result(step_size, radius)

# data, labels = load_samples('../data/samples_Midwest.json', standardize=True)
#
# n = data.shape[0]
# for i in range(n):
#     data[i] = grid(data[i], n=100)
# print('Data preprocessed.')
# print(np.unique(data))
#
# pca_show_figure(data, labels)
#
# print('Grouping')
# grouped_data, grouped_labels = grouping(data, labels, step_size, radius)
# print('Divided into {} Groups'.format(len(grouped_labels)))
# for i, g in enumerate(grouped_labels):
#     print('Group {}: {} data points.'.format(i, len(g)))
# exit()
# with open('../data/grouped_data.json', 'w') as f:
#     json.dump({'data': np.array(grouped_data[0]).tolist(),
#                'labels': np.array(grouped_labels[0]).tolist(),
#                }, f)
#
# with open('../data/grouped_data.json', 'r') as f:
#     json_data = json.load(f)
#     data = json_data['data']
#     labels = json_data['labels']
# data = feature_standardize(data)
# pca_show_figure(data[::-1], labels[::-1])
# cpca = CPCA()
# cpca.fit(data, labels, standardized=False)
# alphas = [0, 0.2, 0.5, 1, 5, 10, 20, 30, 50, 100, 200, 400, 600, 800, 1000]
#
# for alpha in alphas:
#     projected_data, cp = cpca.transform(alpha=alpha)
#     show_figure(projected_data[::-1], labels[::-1], title='α = {}'.format(alpha))
