import json
import os

from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np

from FLHeteroBackend import settings
from pca import get_pcs, get_cluster_list, CPCA
from pca.cluster import clustering_history
from . import RunningState
from utils import load_history, load_samples, load_outputs, load_weights

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
        data = {'datasetNames': 'mnist'}
        return JsonResponse(data)
    if request.method == 'POST':
        request_data = json.loads(request.body)
        dataset_name = request_data['datasetName']

        history = load_history(dataset_name)
        client_names = history['client_names']
        n_clients = client_names.shape[0]
        n_rounds = history['loss'].shape[1]

        rs.add('dataset', dataset_name)
        rs.add('client_names', client_names)
        rs.add('n_clients', n_clients)
        rs.add('n_rounds', n_rounds)

        data = {'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'dimensions': 784,
                'numberOfClients': n_clients,
                'clientNames': client_names.tolist(),
                'trainingDataSize': 4000,
                'testDataSize': 2000,
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

        rs.add('client', client_name)
        rs.add('annotations', [])

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
        weight = load_weights(dataset_name=rs.state['dataset'], client_name=rs.state['client'],
                              n_rounds=rs.state['n_rounds'])
        weight_0 = weight['weights_0']
        weights_server = weight['weights_server']
        weights_client = weight['weights_client']
        weight_0, weights_server, weights_client, idx = clustering_history(weight_0=weight_0,
                                                                           weights_server=weights_server,
                                                                           weights_client=weights_client)
        data = {'weight0': weight_0.tolist(),
                'serverWeights': weights_server.tolist(),
                'clientWeights': weights_client.tolist(),
                'splitPoints': idx.tolist(),
                }
        return JsonResponse(data)


# path('sampling/', views.sampling),
@csrf_exempt
def sampling(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        sampling_type = request_data['samplingType']

        samples, ground_truth = load_samples(datasets=rs.state['dataset'], client_name=rs.state['client'],
                                             sampling_type=sampling_type)

        rs.add('data', samples)
        rs.add('ground_truth', ground_truth)

        data = {'data': samples.tolist()}
        return JsonResponse(data)


# path('pca/', views.pca),
@csrf_exempt
def pca(request):
    if request.method == 'GET':
        pc1, pc2 = get_pcs(rs.state['data'])
        data = {'pc1': pc1.tolist(), 'pc2': pc2.tolist()}
        return JsonResponse(data)


# path('labels/', views.labels),
@csrf_exempt
def labels(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        cm_round = request_data['round']

        output_labels = load_outputs(datasets=rs.state['dataset'], client_name=rs.state['client'], cm_round=cm_round)
        rs.add_dict(output_labels)
        rs.add('cm_round', cm_round)

        outputs_server = rs.state['outputs_server']
        outputs_client = rs.state['outputs_client']
        ground_truth = rs.state['ground_truth']

        data = {'consistencyLabel': (outputs_client == outputs_server).tolist(),
                'groundTruthLabel': ground_truth.tolist(),
                'outputLabel': outputs_server.tolist(),
                }
        return JsonResponse(data)


# path('cluster/', views.cluster),
@csrf_exempt
def cluster(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        n_clusters = 20
        if 'nrOfClusters' in request_data.keys():
            n_clusters = request_data['nrOfClusters']
        cluster_list = get_cluster_list(n_clusters=n_clusters, data=rs.state['data'],
                                        ground_truth=rs.state['ground_truth'],
                                        outputs_server=rs.state['outputs_server'],
                                        outputs_client=rs.state['outputs_client'])
        rs.add('clusters', cluster_list)
        data = {'nrOfClusters': n_clusters,
                'clusterList': cluster_list}
        return JsonResponse(data)


# path('cpca/', views.cpca),
@csrf_exempt
def cpca(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        alpha = settings.DEFAULT_ALPHA
        if 'alpha' in request_data.keys():
            alpha = request_data['alpha']
        cluster_id = request_data['clusterID']
        data = rs.state['data']
        hetero_idx = rs.state['clusters'][cluster_id]['heteroIndex']
        homo_idx = rs.state['outputs_server'] == rs.state['outputs_client']
        bg = data[hetero_idx]
        fg = np.concatenate((data[homo_idx], bg))
        cPCA = CPCA(n_components=2)
        cPCA.fit_transform(target=fg, background=bg, alpha=alpha)

        data = {'cpc1': cPCA.components_[0].tolist(),
                'cpc2': cPCA.components_[1].tolist()}

        return JsonResponse(data)


# path('annotation/', views.annotation),
@csrf_exempt
def annotation(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        cluster_id = request_data['clusterID']
        text = request_data['text']
        data_id = rs.state['clusters'][cluster_id]['heteroIndex']
        rs.state['annotations'].append({'round': rs.state['cm_round'],
                                        'text': text,
                                        'dataIndex': data_id})
        return HttpResponse()


# path('annotationList/', views.annotation_list),
@csrf_exempt
def annotation_list(request):
    if request.method == 'GET':
        return JsonResponse({'annotationList': rs.state['annotations']})
