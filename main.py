# Load the alexandria, model
import mylib.datasets.alexandria as alex
import mylib.models.basic_gnn as gnn
import torch as tc
import torch.nn as nn
from torch.optim import SGD
import wandb

# Init the wandb
run = wandb.init(
    entity="aylwin-johns-hopkins-university",
    project="matbench",
)
# check device
print("CUDA available: ", tc.cuda.is_available())
print("CUDA device count: ", tc.cuda.device_count())

# Load the dataset
dataset = alex.Dataset(root="/home/gengyao/matbench", partial=(0.0,0.05), tensorize_work=True)
# Initialize the model
model = gnn.Model(device='cuda')
print(dataset, model)

optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
loss_fn = nn.L1Loss()
batch_size = 1
batch_size_cnt = 0
loss = 0
loss_sum = 0
# training loop
for epoch in range(100):
    for sample in dataset:
        y_bar = tc.tensor([[sample[0], sample[1]]]).to(model.device)
        y = model.forward(sample[3], sample[4], sample[5])
        loss += loss_fn(y, y_bar)
        batch_size_cnt += 1
        if batch_size_cnt == batch_size:
            # backpropagation
            print(f"epoch:{epoch}, loss:{loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss = 0
            batch_size_cnt = 0
    run.log({"epoch": epoch, "loss": loss_sum})