import json
import numpy as np
from FLHetero import temporal_segment, get_segmented_history
from fedlearn import predict
from pca import get_pca_result
from utils import load_history


def test_initialize():
    history = load_history()
    fed_weight = history['federated']['weight']
    segmentation = temporal_segment(fed_weight, segment_number=20, max_length=5)
    segmented_history = get_segmented_history(history, segmentation)
    data = {'time': segmentation,
            'federated': segmented_history['federated'],
            'others': segmented_history['others'],
            }
    with open('test_data.json', 'w') as f:
        json.dump(data, f)


def test_identify():
    time = 20
    client_name = 'Client-South'
    step_size = 1
    predict(client_name, time)
    hetero_list = get_pca_result(step_size, radius=10)
    data = {'heteroList': hetero_list}
    with open('test_data.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    test_identify()
    # with open('test_data.json', 'r') as f:
    #     data = json.load(f)
    # hl = data['heteroList']
    # print(len(hl))
    # print(hl[0])
