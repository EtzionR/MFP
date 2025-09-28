# GNN object (with 2 GCN layers)
#######################################################################################################
# created by Etzion Harari | TAU
# https://github.com/EtzionR

# imports
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch

# numpy to torch covert function
to_torch = lambda values: values if isinstance(values, torch.Tensor) else torch.from_numpy(values)

# set device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GNN object
class GNN(torch.nn.Module):
    """
    simple GNN object (torch-geometric base, in sklearn style)
    """
    def __init__(self, input_size, output_size, hidden_channels=16, epochs=50, lr=.05, use_tqdm=True):
        """
        init object

        :input_size: number of input dim/number of input features
        :output_size: number of labels for classifiaction
        :hidden_channels: hidden size for GCN layer (int, default: 16)
        :epochs: number of epochs for GNN training (int, default: 50)
        :lr: learning rate for the GNN (float, default: .05)
        :use_tqdm: use tqdm progress bar while training (bool, default: True)
        """
        super().__init__()

        self.lr = lr
        self.progress = tqdm if use_tqdm else lambda x: x

        self.loss = []
        self.epochs = epochs

        self.conv1 = GCNConv(input_size, hidden_channels).to(DEVICE)
        self.conv2 = GCNConv(hidden_channels, output_size).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        """
        forward pass function

        """
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def fit(self ,features, edges, labels, train_index=None):
        """
        GNN training function

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        :labels: node labels (Y, size |V|, torch.tensor/numpy array)
        :train_index: node labels (size |V|, torch.tensor/numpy array/None, default: None)
        """

        # define train indices
        train_index = torch.tensor([True]*len(labels)) if type(train_index)==type(None) else train_index

        # convert all input to toech tensors
        train_index = to_torch(train_index).to(DEVICE)
        features    = to_torch(features).to(DEVICE)
        edges       = to_torch(edges).to(DEVICE)
        labels      = to_torch(labels).to(DEVICE)

        # GNN training
        for _ in self.progress(range(self.epochs)):

            # train
            self.train()
            self.optimizer.zero_grad()

            # forward pass + loss calculation
            out = self(features ,edges)
            loss = self.criterion(out[train_index], labels[train_index])

            # backprop
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, features, edges):
        """
        get prediction from the GNN

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        """
        features = to_torch(features).to(DEVICE)
        edges = to_torch(edges).to(DEVICE)

        self.eval()

        return self(features ,edges).argmax(dim=1)

    def metrics(self, features, edges, labels, test_index, average='weighted'):
        """
        test performance (ACCURACY)

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        :labels: node labels (Y, size |V|, torch.tensor/numpy array)
        :train_index: node labels (size |V|, torch.tensor/numpy array/None, default: None)
        """

        # define train indices
        test_index = torch.tensor([True] * len(labels)) if type(test_index) == type(None) else test_index

        # convert all input to toech tensors
        test_index = to_torch(test_index).to(DEVICE)
        features = to_torch(features).to(DEVICE)
        edges = to_torch(edges).to(DEVICE)
        labels = to_torch(labels).to(DEVICE)

        pred = self.predict(features, edges)

        pred_test = pred[test_index].detach().cpu().numpy()
        true_test = labels[test_index].detach().cpu().numpy()

        return {'Accuracy':  accuracy_score(pred_test, true_test),
                'Recall':    recall_score(pred_test, true_test, average=average),
                'Precision': precision_score(pred_test, true_test, average=average),
                'F1':        f1_score(pred_test, true_test, average=average)}

    def test(self, features, edges, labels, test_index):
        """
        test performance (ACCURACY)

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        :labels: node labels (Y, size |V|, torch.tensor/numpy array)
        :train_index: node labels (size |V|, torch.tensor/numpy array/None, default: None)
        """

        # define train indices
        test_index = torch.tensor([True]*len(labels)) if type(test_index)==type(None) else test_index

        # convert all input to toech tensors
        test_index = to_torch(test_index).to(DEVICE)
        features    = to_torch(features).to(DEVICE)
        edges       = to_torch(edges).to(DEVICE)
        labels      = to_torch(labels).to(DEVICE)

        pred = self.predict(features, edges)


        return ((pred[test_index] == labels[test_index]).sum( ) /test_index.sum()).item()


class GNN_Regressor(torch.nn.Module):
    """
    simple GNN object (torch-geometric base, in sklearn style)
    """

    def __init__(self, input_size, output_size=1, hidden_channels=16, epochs=50, lr=.05, use_tqdm=True):
        """
        init object

        :input_size: number of input dim/number of input features
        :output_size: number of labels for classifiaction
        :hidden_channels: hidden size for GCN layer (int, default: 16)
        :epochs: number of epochs for GNN training (int, default: 50)
        :lr: learning rate for the GNN (float, default: .05)
        :use_tqdm: use tqdm progress bar while training (bool, default: True)
        """
        super().__init__()

        self.lr = lr
        self.progress = tqdm if use_tqdm else lambda x: x

        self.loss = []
        self.epochs = epochs

        self.conv1 = GCNConv(input_size, hidden_channels).to(DEVICE)
        self.conv2 = GCNConv(hidden_channels, output_size).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, edge_index):
        """
        forward pass function

        """
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def fit(self, features, edges, labels, train_index=None):
        """
        GNN training function

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        :labels: node labels (Y, size |V|, torch.tensor/numpy array)
        :train_index: node labels (size |V|, torch.tensor/numpy array/None, default: None)
        """

        # define train indices
        train_index = torch.tensor([True] * len(labels)) if type(train_index) == type(None) else train_index

        # convert all input to toech tensors
        train_index = to_torch(train_index).to(DEVICE)
        features = to_torch(features).to(DEVICE)
        edges = to_torch(edges).to(DEVICE)
        labels = to_torch(labels).to(DEVICE)

        # GNN training
        for _ in self.progress(range(self.epochs)):
            # train
            self.train()
            self.optimizer.zero_grad()

            # forward pass + loss calculation
            out = self(features, edges)
            loss = self.criterion(out[train_index], labels[train_index])

            # backprop
            loss.backward()
            self.optimizer.step()

            self.loss.append(loss.item())

        return self

    def predict(self, features, edges):
        """
        get prediction from the GNN

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        """
        features = to_torch(features).to(DEVICE)
        edges = to_torch(edges).to(DEVICE)

        self.eval()

        return self(features, edges).reshape(-1).detach()

    def test(self, features, edges, labels, test_index):
        """
        test performance (ACCURACY)

        :features: nodes feature matrix (X, size |V|*|d|, torch.tensor/numpy array)
        :edges: edge matrix (E, size |2|*|E|, torch.tensor/numpy array)
        :labels: node labels (Y, size |V|, torch.tensor/numpy array)
        :train_index: node labels (size |V|, torch.tensor/numpy array/None, default: None)
        """

        # define train indices
        test_index = torch.tensor([True] * len(labels)) if type(test_index) == type(None) else test_index

        # convert all input to toech tensors
        test_index = to_torch(test_index).to(DEVICE)
        features = to_torch(features).to(DEVICE)
        edges = to_torch(edges).to(DEVICE)
        labels = to_torch(labels).to(DEVICE)

        pred = self.predict(features, edges)
        return torch.mean(torch.pow(pred[test_index] - labels[test_index], 2)).item()

#######################################################################################################
# created by Etzion Harari | TAU
# https://github.com/EtzionR