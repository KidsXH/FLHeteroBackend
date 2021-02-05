import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from fedlearn import predict
from pca import get_pca_result
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
        request_data = json.loads(request.body)
        # print(request_data)
        time = request_data['time']
        client_name = request_data['client']
        step_size = request_data['step']
        predict(client_name, time)
        hetero_list, pca, labels_het, fed_result = get_pca_result(step_size, radius=10)
        data = {
            'heteroList': hetero_list,
            'pca': pca,
            'heteroLabels': labels_het,
            'fedResult': fed_result,
        }
        return JsonResponse(data)


@csrf_exempt
def customize(request):
    if request.method == 'POST':
        data = {}
        return JsonResponse(data)
