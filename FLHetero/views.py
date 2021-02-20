import json
import os

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from FLHeteroBackend import settings
from fedlearn import predict_mnist
from pca import get_pca_result, load_samples
from . import start_federated_learning, temporal_segment, get_segmented_history
from utils import load_history


def hello(request):
    return JsonResponse('Hello', safe=False)


def initialize(request):
    if request.method == 'GET':
        start_federated_learning()
        history = load_history()
        fed_weight = history['federated']['weight']
        segmentation = temporal_segment(fed_weight, segment_number=20, max_length=5)
        segmented_history = get_segmented_history(history, segmentation)
        data = {'time': segmentation,
                'federated': segmented_history['federated'],
                'others': segmented_history['others'],
                }
        return JsonResponse(data)


@csrf_exempt
def identify(request):
    if request.method == 'POST':
        # print(request.body)
        request_data = json.loads(request.body)
        # print(request_data)
        cm_round = request_data['round']
        client_name = request_data['client']
        predict_mnist(client_name, cm_round)
        samples, local_data, ground_truth, outputs_client, outputs_server = load_samples(os.path.join(settings.DATA_DIR,
                                                                                                      'samples.json'))
        hetero_list, pca = get_pca_result(n_clusters=5,
                                          data=samples,
                                          ground_truth=ground_truth,
                                          outputs_client=outputs_client,
                                          outputs_server=outputs_server)
        data = {'localData': local_data.tolist(),
                'samples': samples.tolist(),
                'groundTruth': ground_truth.tolist(),
                'outputLabels': outputs_server.tolist(),
                'heteroLabels': (outputs_client != outputs_server).tolist(),
                'fedResult': (outputs_server == ground_truth).tolist(),
                'pca': pca,
                'heteroList': hetero_list,
                }
        return JsonResponse(data)


@csrf_exempt
def customize(request):
    if request.method == 'POST':
        data = {}
        return JsonResponse(data)
