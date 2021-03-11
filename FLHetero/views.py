import json
import os

from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np

from FLHeteroBackend import settings
from pca import get_cluster_list, CPCA, pca_weights
from . import RunningState
from utils import load_history, load_samples, load_outputs, load_weights, sample_weight

rs = RunningState()


def hello(request):
    return JsonResponse('Hello', safe=False)


@csrf_exempt
def customize(request):
    if request.method == 'POST':
        data = {}
        return JsonResponse(data)


# path('datasets/', views.datasets),
@csrf_exempt
def datasets(request):
    if request.method == 'GET':
        data = {'datasetNames': 'mnist_mlp'}
        return JsonResponse(data)
    if request.method == 'POST':
        request_data = json.loads(request.body)
        dataset_name = request_data['datasetName']

        history = load_history(dataset_name)
        client_names = history['client_names']
        n_clients = client_names.shape[0]
        n_rounds = history['loss'].shape[1]

        rs.set('dataset', dataset_name)
        rs.set('client_names', client_names)
        rs.set('n_clients', n_clients)
        rs.set('n_rounds', n_rounds)

        data = {'labels': 'The value of the handwritten digit.',
                'dimensions': 784,
                'numberOfClients': n_clients,
                'clientNames': client_names.tolist(),
                'trainingDataSize': 5400,
                'testDataSize': 600,
                'communicationRounds': n_rounds,
                }

        # for (k, v) in data.items():
        #     print('{}: {} {}'.format(k, type(v), np.shape(v)))

        return JsonResponse(data)


# path('client/', views.client),
@csrf_exempt
def client(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        client_name = request_data['clientName']

        if client_name not in rs.state['client_names']:
            return HttpResponseBadRequest

        rs.set('client', client_name)
        rs.set('annotations', [])

        client_idx = np.where(rs.state['client_names'] == client_name)

        history = load_history(rs.state['dataset'])
        loss = history['loss'][client_idx][0]
        val_acc = history['val_acc'][client_idx][0]

        data = {
            'loss': loss.tolist(),
            'valAcc': val_acc.tolist(),
        }
        return JsonResponse(data)


# path('weights/', views.weights),
@csrf_exempt
def weights(request):
    if request.method == 'GET':
        weights_0, weights_client, weights_server, cosines = load_weights(dataset_name=rs.state['dataset'],
                                                                          client_name=rs.state['client'],
                                                                          n_rounds=rs.state['n_rounds'])
        # print(weights_0, weights_client, weights_server, cosines)

        weights_0, weights_client, weights_server = sample_weight(weights_0, weights_client, weights_server, num=1000)
        weights_0, weights_client, weights_server = pca_weights(weights_0, weights_client, weights_server)

        data = {'weight0': weights_0.tolist(),
                'serverWeights': weights_server.tolist(),
                'clientWeights': weights_client.tolist(),
                'cosines': cosines.tolist(),
                }

        return JsonResponse(data)


# path('sampling/', views.sampling),
@csrf_exempt
def sampling(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        sampling_type = request_data['samplingType']
        cm_round = request_data['round']

        samples, ground_truth = load_samples(datasets=rs.state['dataset'], client_name=rs.state['client'],
                                             sampling_type=sampling_type)

        output_labels = load_outputs(datasets=rs.state['dataset'], client_name=rs.state['client'], cm_round=cm_round,
                                     sampling_type=rs.state['sampling_type'])
        outputs_server = rs.state['outputs_server']  # type: np.ndarray
        outputs_client = rs.state['outputs_client']  # type: np.ndarray

        rs.set('data', samples)
        rs.set('sampling_type', sampling_type)
        rs.set('ground_truth', ground_truth)
        rs.add_dict(output_labels)
        rs.set('cm_round', cm_round)

        # samples = np.round(samples.astype(float), 5)
        data = {'data': samples.tolist(),
                'consistencyLabel': (outputs_client == outputs_server).tolist(),
                'groundTruthLabel': ground_truth.tolist(),
                'outputLabel': outputs_server.tolist(),
                'localOutputLabel': outputs_client.tolist(),
                }
        return JsonResponse(data)


# path('cpca/all', views.cpca_all),
@csrf_exempt
def cpca_all(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        alpha = None
        if 'alpha' in request_data.keys():
            alpha = request_data['alpha']

        cPCA = CPCA(n_components=2)
        data = rs.state['data']
        hetero_labels = rs.state['outputs_server'] != rs.state['outputs_client']
        cPCA.fit_transform(target=data, background=data[hetero_labels], alpha=alpha)

        data = {'alpha': cPCA.alpha,
                'cpc1': cPCA.components_[0].tolist(),
                'cpc2': cPCA.components_[1].tolist()}

        rs.set('cpca_all_result', data)

        return JsonResponse(data)


# path('cluster/', views.cluster),
@csrf_exempt
def cluster(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        n_clusters = None
        if 'nrOfClusters' in request_data.keys():
            n_clusters = request_data['nrOfClusters']

        cluster_list = get_cluster_list(n_clusters=n_clusters, client_name=rs.state['client'],
                                        data=rs.state['data'], sampling_type=rs.state['sampling_type'],
                                        outputs_server=rs.state['outputs_server'],
                                        outputs_client=rs.state['outputs_client'])

        rs.set('clusters', cluster_list)

        data = {'nrOfClusters': len(cluster_list),
                'clusterList': cluster_list}
        return JsonResponse(data)


# path('cpca/cluster', views.cpca_cluster),
@csrf_exempt
def cpca_cluster(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        # print(request_data)
        alpha = None
        if 'alpha' in request_data.keys():
            alpha = request_data['alpha']

        cluster_id = request_data['clusterID']

        data = rs.state['data']
        hetero_idx = rs.state['clusters'][cluster_id]['heteroIndex']
        homo_idx = rs.state['outputs_server'] == rs.state['outputs_client']

        bg = data[hetero_idx]

        if bg.shape[0] == 1:
            return JsonResponse(rs.state['cpca_all_result'])

        fg = np.concatenate((data[homo_idx], bg))

        # print('fg: {}, bg: {}, alpha: {}'.format(fg.shape, bg.shape, alpha))

        cPCA = CPCA(n_components=2)
        cPCA.fit_transform(target=fg, background=bg, alpha=alpha)

        data = {'alpha': cPCA.alpha,
                'cpc1': cPCA.components_[0].tolist(),
                'cpc2': cPCA.components_[1].tolist()}

        return JsonResponse(data)


# path('annotation/', views.annotation),
@csrf_exempt
def annotation(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        data_index = request_data['dataIndex']
        text = request_data['text']
        rs.state['annotations'].append({'round': rs.state['cm_round'],
                                        'text': text,
                                        'dataIndex': data_index})
        return HttpResponse()


# path('annotationList/', views.annotation_list),
@csrf_exempt
def annotation_list(request):
    if request.method == 'GET':
        return JsonResponse({'annotationList': rs.state['annotations']})
