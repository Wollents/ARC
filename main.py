import argparse
from utils import *
import warnings
from train_test import ARCDetector
import numpy as np
from torch_geometric.nn import Node2Vec, GCN

from dataloader import DataLoader


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def test_dataloader(data: Data):
    loader = DataLoader(
        data,
        num_neighbors=[10] * 2,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        p=1.0,
        q=1.0,
        batch_size=128)
    # TODO: 注意sample_data会有很多节点，但是只需要batch_size的长度，因此最后的embedding只取最后的batch_size的切片
    # TODO： 具体参考DOMINANT pygod实现
    for sample_data, pos_rw, neg_rw in loader:
        print(sample_data)
        print(pos_rw)
        print(neg_rw)


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--model', type=str, default='ARC')
parser.add_argument('--shot', type=int, default=10)
parser.add_argument('--json_dir', type=str, default='./params')
args = parser.parse_known_args()[0]

datasets_test = ['cora', 'citeseer', 'ACM', 'BlogCatalog',
                 'Facebook', 'weibo', 'Reddit', 'Amazon', 'cs', 'photo', 'tolokers']
datasets_train = ['pubmed', 'Flickr', 'questions', 'YelpChi']

model = args.model
model_result = {'name': model}
print('Training on {} datasets:'.format(len(datasets_train)), datasets_train)
print('Test on {} datasets:'.format(len(datasets_test)), datasets_test)

train_config = {
    'device': 'cuda:0',
    'epochs': 40,
    'testdsets': datasets_test,
}

dims = 64
data_train = [Dataset(dims, name) for name in datasets_train]
data_test = [Dataset(dims, name) for name in datasets_test]  # CPU
model_config = read_json(model, args.shot, args.json_dir)
test_dataloader(data_test[0].graph)

if model_config is None:
    model_config = {
        "model": "ARC",
        "lr": 1e-5,
        "drop_rate": 0,
        "h_feats": 1024,
        "num_prompt": 10,
        "num_hops": 2,
        "weight_decay": 5e-5,
        "in_feats": 64,
        "num_layers": 4,
        "activation": "ELU"
    }
    print('use default model config')
else:
    print('use saved best model config')
    print(model_config)

for tr_data in data_train:
    tr_data.propagated(model_config['num_hops'])
for te_data in data_test:
    te_data.propagated(model_config['num_hops'])

model_config['model'] = model
model_config['in_feats'] = dims
# Initialize dictionaries to store scores for each test dataset
auc_dict = {}
pre_dict = {}
for t in range(args.trials):
    seed = t
    set_seed(seed)
    print("Model {}, Trial {}".format(model, seed))
    train_config['seed'] = seed
    for te_data in data_test:
        te_data.few_shot(args.shot)
    data = {'train': data_train, 'test': data_test}
    detector = ARCDetector(train_config, model_config, data)
    test_score_list = detector.train()
    # Aggregate scores for each test dataset
    for test_data_name, test_score in test_score_list.items():
        if test_data_name not in auc_dict:
            auc_dict[test_data_name] = []
            pre_dict[test_data_name] = []
        auc_dict[test_data_name].append(test_score['AUROC'])
        pre_dict[test_data_name].append(test_score['AUPRC'])
        print(f'Test on {test_data_name}, AUC is {auc_dict[test_data_name]}')

# Calculate mean and standard deviation for each test dataset
auc_mean_dict, auc_std_dict, pre_mean_dict, pre_std_dict = {}, {}, {}, {}

for test_data_name in auc_dict:
    auc_mean_dict[test_data_name] = np.mean(auc_dict[test_data_name])
    auc_std_dict[test_data_name] = np.std(auc_dict[test_data_name])
    pre_mean_dict[test_data_name] = np.mean(pre_dict[test_data_name])
    pre_std_dict[test_data_name] = np.std(pre_dict[test_data_name])
# Output the results for each test dataset
for test_data_name in auc_mean_dict:
    str_result = 'AUROC:{:.4f}+-{:.4f}, AUPRC:{:.4f}+-{:.4f}'.format(
        auc_mean_dict[test_data_name],
        auc_std_dict[test_data_name],
        pre_mean_dict[test_data_name],
        pre_std_dict[test_data_name])
    print('-' * 50 + test_data_name + '-' * 50)
    print('str_result', str_result)
