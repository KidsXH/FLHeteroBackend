import json
import os

import numpy as np
import torch

from fedlearn.model import Net
from fedlearn.datasets import get_region_data_loader
from FLHeteroBackend import settings

ckpt_path = os.path.join(settings.DATA_DIR, 'checkpoint')


def predict(client_name, time):
    server_ckpt_path = os.path.join(ckpt_path, 'Server', 'Server_r{}.cp'.format(time))
    client_ckpt_path = os.path.join(ckpt_path, client_name, '{}_r{}.cp'.format(client_name, time))

    model_s = Net().cuda()
    model_s.load_state_dict(torch.load(server_ckpt_path))
    model_c = Net().cuda()
    model_c.load_state_dict(torch.load(client_ckpt_path))

    test_loader = get_client_data_loader(client_name)

    good_samples = []
    bad_samples = []
    correct_s = 0
    correct_c = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            outputs_s = model_s(inputs)
            _, predicted_s = torch.max(outputs_s.data, 1)
            outputs_c = model_c(inputs)
            _, predicted_c = torch.max(outputs_c.data, 1)
            total += labels.size(0)
            correct_s += (predicted_s == labels).sum().item()
            correct_c += (predicted_c == labels).sum().item()
            good_samples.append(inputs[predicted_c == predicted_s].cpu().numpy())
            bad_samples.append(inputs[predicted_c != predicted_s].cpu().numpy())

    # print('Acc_server: {} Acc_client: {}'.format(correct_s / total, correct_c / total))
    good_samples = np.concatenate(good_samples)
    bad_samples = np.concatenate(bad_samples)
    # print('Good Samples {}, Bad Samples: {}'.format(good_samples.shape, bad_samples.shape))

    syn_data = {
        'positive': good_samples.tolist(),
        'negative': bad_samples.tolist(),
    }

    with open('data/samples.json', 'w') as f:
        json.dump(syn_data, f)


def get_client_data_loader(client_name):
    return get_region_data_loader(path=os.path.join(settings.DATA_DIR, 'datasets'), region=client_name[7:],
                                  batch_size=100)
