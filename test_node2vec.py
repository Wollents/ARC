from torch_geometric.nn import Node2Vec, GCN
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score
dataset = Dataset(64, "cora")
data = dataset.graph

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True).to(device)

num_workers = 4
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc

@torch.no_grad()
def test_anoamly():
    model.eval()
    z = model()
    s_ = torch.sigmoid(z @ z.T).to(device)
    s = data.adj.to(device)
    error = torch.pow(s_ - s, 2)
    score = error.mean(dim=-1)
    auc = roc_auc_score(data.ano_labels.cpu().numpy(), score.cpu().numpy())
    return auc


for epoch in range(1, 101):
    torch.cuda.reset_peak_memory_stats()
    loss = train()
    train_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    auc = test_anoamly()
    test_men = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Auc: {auc:.4f}, "Train_mem : {train_mem:.4f} MB", "Test_mem : {test_men:.4f} MB", ')
