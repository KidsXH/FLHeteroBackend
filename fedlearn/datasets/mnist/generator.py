from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
from FLHeteroBackend import settings
import json
import os

DATA_HOME = settings.DATA_HOME['mnist']
TRAIN_FILE = os.path.join(DATA_HOME, 'train.json')
TEST_FILE = os.path.join(DATA_HOME, 'test.json')

groups_per_class = 2
samples_per_group = 3000
random_seed = 123

"""Data Structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
"""

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml("mnist_784", data_home=DATA_HOME)
# mu = np.mean(mnist.data.astype(np.float32), 0)
# sigma = np.std(mnist.data.astype(np.float32), 0)
# mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)
mnist_data = []

for i in range(10):
    idx = mnist.target == str(i)
    mnist_data.append(mnist.data[idx])
    mnist_data[-1] = mnist_data[-1][:samples_per_group * groups_per_class]

data = []
target = []

seq_a = np.arange(0, 10, 1)

while True:
    seq_b = np.random.permutation(10)
    if (seq_a == seq_b).sum() == 0:
        break

for a, b in zip(seq_a, seq_b):
    x = np.concatenate((mnist_data[a][:samples_per_group], mnist_data[b][samples_per_group:]))
    y = np.array([a] * samples_per_group + [b] * samples_per_group, dtype=int)
    data.append(x)
    target.append(y)

train_data = {'username': [], 'data': [], 'target': []}
test_data = {'username': [], 'data': [], 'target': []}

for uid in trange(10):
    username = 'Client-{}'.format(uid)
    x = data[uid]
    y = target[uid]
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

    n_samples = x.shape[0]
    n_train = int(n_samples * 0.9)

    train_data['username'].append(username)
    train_data['data'].append(x[:n_train].tolist())
    train_data['target'].append(y[:n_train].tolist())

    test_data['username'].append(username)
    test_data['data'].append(x[n_train:].tolist())
    test_data['target'].append(y[n_train:].tolist())

print('Train Data: {}, {}'.format(np.shape(train_data['data']), np.shape(train_data['target'])))
print('Test Data: {}, {}'.format(np.shape(test_data['data']), np.shape(test_data['target'])))

with open(TRAIN_FILE, 'w') as f:
    json.dump(train_data, f)

with open(TEST_FILE, 'w') as f:
    json.dump(test_data, f)
