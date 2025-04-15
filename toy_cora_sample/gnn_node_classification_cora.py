import torch
import torch.nn.functional as F
import datetime # Added for timestamping log
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 1. Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Check if MPS is available and use it, otherwise use CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

data = data.to(device)

# 2. Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN(dataset.num_node_features, dataset.num_classes).to(device)

# 3. Set up optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# 4. Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 5. Evaluation function
def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

print("Starting training...")
for epoch in range(1, 101): # Train for 100 epochs for a quick example
    loss = train()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

print("Training finished.")

# 6. Final evaluation and logging
train_acc, val_acc, test_acc = test()
print(f'Final Train Accuracy: {train_acc:.4f}')
print(f'Final Validation Accuracy: {val_acc:.4f}')
print(f'Final Test Accuracy: {test_acc:.4f}')

# Log performance
log_file = "performance_log.txt"
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = f"{now} - Final Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}\n"

print(f"Appending performance to {log_file}")
with open(log_file, "a") as f:
    f.write(log_entry)

# Example usage:
# You might need to install torch_geometric first:
# pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.3.0+${CUDA}.html
# (Replace ${CUDA} with your CUDA version like cpu, cu118, cu121, etc.)
# Then run the script: python gnn_node_classification.py 