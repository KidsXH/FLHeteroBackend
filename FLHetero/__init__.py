import numpy as np
from utils import chebyshev_distance

MAX_STEP = 100


def start_federated_learning():
    pass


def avg_weight(weights):
    sum_weight = weights[0]
    for w in weights[1:]:
        sum_weight += w
    return sum_weight / len(weights)


def seg_wgl(history, segmentation):
    h_weight = np.array(history['weight'])
    h_loss = history['loss']
    weight = []
    grad = []
    loss = []
    for idx, (l, r) in enumerate(segmentation):
        weight.append(avg_weight(h_weight[l: r]))
        grad.append(np.zeros(len(weight[-1]), dtype=float) if idx == 0 else (weight[-1] - weight[-2]))
        loss.append(np.mean(h_loss[l: r]))
    weight = np.array(weight).tolist()
    grad = np.array(grad).tolist()
    loss = np.array(loss).tolist()
    return weight, grad, loss


def get_segmented_history(history, segmentation):
    fw, fg, fl = seg_wgl(history['federated'], segmentation)
    result = {
        'time': segmentation,
        'federated': {
            'weight': fw,
            'gradient': fg,
            'loss': fl,
        },
        'others': []
    }
    for his in history['others']:
        ow, og, ol = seg_wgl(his, segmentation)
        result['others'].append({
            'clientName': his['clientName'],
            'weight': ow,
            'gradient': og,
            'loss': ol,
        })

    return result


def temporal_segment(data, segment_number, max_length):
    data = np.array(data)
    T, n = data.shape

    if T < segment_number:
        return data

    low, high = 0, np.max(data) - np.min(data)
    step = 0
    while low < high:
        step += 1
        mid = (low + high) / 2
        result = []
        head = 0
        for i in range(1, T):
            if i - head == max_length or chebyshev_distance(data[head], data[i]) > mid:
                result.append((head, i))
                head = i
        result.append((head, T))

        len_r = len(result)

        if len_r == segment_number or step == MAX_STEP:
            return result
        elif len_r > segment_number:
            low = mid
        else:
            high = mid
