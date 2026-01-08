from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import select_class_set


class Hebb:
    def __init__(self, n_input: int, n_output: int, lr=0.0001):
        """
        Hebbian attractor class
        :param n_input: input size
        :param n_output: output size
        :param lr: adaptation step size
        """
        self.output_weights = np.zeros((n_output, n_input))

        self.lr = lr
        self.li = 0
        self.A = np.zeros(n_input*n_output)
        self.B = np.zeros(n_input*n_output)
        self.C = np.zeros(n_input*n_output)
        self.D = np.zeros(n_input*n_output)

    def hebbian_update(self, pre_synaptic: np.ndarray, pos_synaptic: np.ndarray):
        """
        Set weights of NN.

        :param pos_synaptic:
        :param pre_synaptic:
        """
        weights_delta = self.lr*(self.A*pre_synaptic +
                                 self.B*pos_synaptic +
                                 self.C*pre_synaptic*pos_synaptic +
                                 self.D)
        self.output_weights += weights_delta.reshape(self.output_weights.shape)
        self.li += 1

    def forward(self, input: np.ndarray):
        """
        Forward pass through the Hebbian network.
        Automatically update weight parameters according to ABCD rules.

        :param input:
        :return: network output
        """
        output_l = np.tanh(np.dot(self.output_weights, input))

        pre_synaptic = np.tile(input, len(output_l))
        pos_synaptic = output_l.repeat(len(input))

        self.hebbian_update(pre_synaptic, pos_synaptic)
        return output_l


class HebbTorch(nn.Module):
    def __init__(self, n_input: int, n_output: int, lr: float = 0.0001, device=None):
        """
        Differentiable Hebbian attractor implemented in PyTorch.

        :param n_input: number of input features
        :param n_output: number of output neurons
        :param lr: Hebbian learning rate (update scale)
        :param device: 'cpu' or 'cuda'
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output weights
        self.register_buffer("output_weights", torch.zeros(n_output, n_input, device=self.device))
        # ABCD rule coefficients
        self.register_buffer("A", torch.zeros(n_output * n_input, device=self.device))
        self.register_buffer("B", torch.zeros(n_output * n_input, device=self.device))
        self.register_buffer("C", torch.zeros(n_output * n_input, device=self.device))
        self.register_buffer("D", torch.zeros(n_output * n_input, device=self.device))

        self.li = 0  # step counter

    @torch.no_grad()
    def hebbian_update(self, pre_synaptic: torch.Tensor, pos_synaptic: torch.Tensor):
        """
        Hebbian update rule: ΔW = lr * (A*x + B*y + C*x*y + D)
        pre_synaptic: flattened input pattern (n_input*n_output)
        pos_synaptic: flattened output pattern (n_input*n_output)
        """
        delta_w = self.lr * (
            self.A * pre_synaptic +
            self.B * pos_synaptic +
            self.C * pre_synaptic * pos_synaptic +
            self.D
        )
        delta_w = delta_w.view(self.n_output, self.n_input)
        self.output_weights += delta_w
        self.li += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Hebbian network with auto-update.

        :param x: input vector, shape [n_input] or [batch, n_input]
        :return: output activations, shape [n_output] or [batch, n_output]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Linear activation + tanh nonlinearity
        y = torch.tanh(F.linear(x, self.output_weights))  # [batch, n_output]

        # Perform Hebbian update (non-differentiable local rule)
        # Each sample independently updates the shared weights
        for i in range(x.size(0)):
            xi = x[i].detach().flatten()
            yi = y[i].detach().flatten()
            pre = xi.repeat_interleave(self.n_output)
            post = yi.repeat(self.n_input)
            self.hebbian_update(pre, post)

        return y


class ConvReservoir(nn.Module):
    def __init__(self, layers, bias=True, device=None):
        super().__init__()
        self.name = "ConvReservoir"
        self.bias = bias
        self.layers_config = layers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_channels = 1
        conv_blocks = []
        for num_filters, fsize in layers[1:]:
            block = nn.Sequential(
                nn.Conv2d(in_channels, num_filters, fsize, stride=1, padding=1, bias=bias),
                nn.GroupNorm(1, num_filters),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            conv_blocks.append(block)
            in_channels = num_filters

        self.layers = nn.Sequential(*conv_blocks)
        self.description = f'ConvNet: {self.layers}'
        self.to(self.device)

    def forward(self, X):
        if X.ndim == 3:
            X = X.unsqueeze(0)
        elif X.ndim != 4:
            raise ValueError(f"Expected input of shape (N,H,W,C), got {X.shape}")
        # Convert (N,H,W,C) → (N,C,H,W)
        if X.shape[1] != 1:  # if channel not first
            X = X.permute(0, 3, 1, 2).contiguous().to(self.device)
        out = self.layers(X)
        return out.flatten(1)


class MLPReservoir(nn.Module):
    def __init__(self, layers, bias=True, device=None):
        super().__init__()
        self.name = 'MLPReservoir'
        self.layers_config = layers
        self.description = f'MLPReservoir: I: {layers[0]} -> Tanh {layers[1:]}'
        self.bias = bias
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # build linear layers with fixed random weights
        linear_layers = []
        for i in range(len(layers) - 1):
            in_f, out_f = layers[i], layers[i + 1]
            layer = nn.Linear(in_f, out_f, bias=bias)
            # initialize with random uniform weights in [-1, 1]
            with torch.no_grad():
                layer.weight.copy_(torch.rand_like(layer.weight)*2-1)
                if bias:
                    layer.bias.copy_(torch.rand_like(layer.bias)*2-1)
            # freeze weights
            for p in layer.parameters():
                p.requires_grad = False
            linear_layers.append(layer)
        self.layers = nn.ModuleList(linear_layers)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through fixed-weight MLP with Tanh activations.
        x: torch.Tensor of shape (..., input_dim)
        returns: torch.Tensor of shape (..., last_layer_dim)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.view(-1, self.layers_config[0])  # flatten input if needed
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class MetaHebbian(nn.Module):
    def __init__(self, network, n_classes, n_shots=10, n_query=15, lr_inner=0.0001, input_size=(1,28,28)):
        super().__init__()
        self.net = network
        self.device = network.device
        self.feature_dim = len(network(torch.randn(input_size).to(self.device)).flatten())
        self.n_classes = n_classes
        self.n_shots = n_shots
        self.n_query = n_query
        self.lr_inner = lr_inner
        self.net.batch_size = n_classes*n_shots
        print(
              f'\n###########  {n_classes}-way:\t{n_shots}-shot'
              f'\n{network.description}\n',
              f'Mu:  \t{lr_inner}\n',
              )
        self.to(self.device)


    def forward(self, train_dataset, val_dataset):
        """
        Perform meta-learning forward pass.
        train_dataset: Data loader for meta learning problem
        """
        # === Construct support and query sets ===
        (support_set, query_set), label_map = self.get_support_and_query(train_dataset)


        # Compute embeddings for few-shot support set
        proto_embeddings = torch.empty(self.n_classes, self.feature_dim, requires_grad=False).to(self.device)
        for c in range(self.n_classes):
            embeddings = torch.empty(self.n_shots, self.feature_dim, requires_grad=False).to(self.device)
            for i_shot in range(self.n_shots):
                img, label = support_set[c*self.n_shots + i_shot]
                emb = self.net(img.to(self.device))
                embeddings[i_shot] = emb
            proto_embeddings[label_map[int(label)]] = torch.mean(embeddings, dim=0)
        proto_mu = torch.mean(proto_embeddings, dim=0)
        proto_vecs = torch.linalg.norm(proto_embeddings - proto_mu, dim=1)
        proto_vecs = (proto_embeddings - proto_mu) / proto_vecs[:, np.newaxis]

        # Initialize Hebbian head
        head = HebbTorch(proto_vecs.shape[1], proto_vecs.shape[0], self.lr_inner, self.device)
        head.A = -torch.flatten(proto_vecs)
        head.D = torch.flatten(proto_vecs) * (1 + 1 / self.n_classes)

        # Inner loop: Hebbian adaptation without labels
        for img, _ in train_dataset:
            emb = self.net(img.to(self.device))
            lateral_inhib = torch.diag(torch.tanh(proto_vecs @ head.output_weights.T)).repeat(proto_vecs.shape[1])
            head.B = lateral_inhib * (1 + 1 / self.n_classes)
            head.C = -lateral_inhib
            head.forward(emb - proto_mu)
            if head.li >= 500:
                break

        # === Query evaluation (differentiable meta loss) ===
        weights = head.output_weights
        imgs = torch.stack([img.to(self.device) for img, _ in query_set])
        labels_q = torch.tensor([label_map[int(label)] for _, label in query_set], device=self.device)

        query_emb = self.net(imgs)
        # Get Hebbian "logits" in torch form
        outputs = query_emb @ weights.T
        # Differentiable loss
        query_loss = F.cross_entropy(outputs, labels_q)
        query_accuracy = (torch.argmax(outputs, dim=1) == labels_q).float().mean()

        # validation_loss, validation_accuracy = self.validation(val_dataset)
        return query_loss, query_accuracy

    def get_support_and_query(self, dataset):
        # Ensure targets are numpy ints
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        unique_labels = np.unique(targets)
        unique_labels = np.random.permutation(unique_labels)

        support_set, query_set = [], []
        for c in unique_labels:
            one_class_set = select_class_set(dataset, [int(c)])  # pass plain int
            indices = np.random.randint(0, len(one_class_set), self.n_shots + self.n_query)
            for i_shot in range(self.n_shots):
                img, label = one_class_set[indices[i_shot]]
                support_set.append((img, label))
            for i_meta in range(self.n_query):
                query_set.append(one_class_set[indices[self.n_shots + i_meta]])

        # Make label_map from plain ints
        label_map = {int(c): i for i, c in enumerate(unique_labels)}
        return (support_set, query_set), label_map

    def validation(self, validation_dataset):
        """
        validation meta-learning forward pass.
        train_dataset: Data for meta learning problem
        test_dataset: Data for meta learning problem
        """
        with torch.no_grad():
            (support_set, _), label_map = self.get_support_and_query(validation_dataset)

            # Compute embeddings for few-shot support set
            proto_embeddings = torch.empty(self.n_classes, self.feature_dim, requires_grad=False).to(self.device)
            for c in range(self.n_classes):
                embeddings = torch.empty(self.n_shots, self.feature_dim, requires_grad=False).to(self.device)
                for i_shot in range(self.n_shots):
                    img, label = support_set[c * self.n_shots + i_shot]
                    emb = self.net(img.to(self.device))
                    embeddings[i_shot] = emb
                proto_embeddings[label_map[int(label)]] = torch.mean(embeddings, dim=0)
            proto_mu = torch.mean(proto_embeddings, dim=0)
            proto_vecs = torch.linalg.norm(proto_embeddings - proto_mu, dim=1)
            proto_vecs = (proto_embeddings - proto_mu) / proto_vecs[:, np.newaxis]

            # Initialize Hebbian network
            head = HebbTorch(proto_vecs.shape[1], proto_vecs.shape[0], self.lr_inner, self.device)
            head.A = -torch.flatten(proto_vecs)
            head.D = torch.flatten(proto_vecs) * (1 + 1 / self.n_classes)

            # Inner Hebbian adaptation without labels
            for img, _ in validation_dataset:
                emb = self.net(img.to(self.device))
                lateral_inhib = torch.diag(torch.tanh(proto_vecs @ head.output_weights.T)).repeat(proto_vecs.shape[1])
                head.B = lateral_inhib * (1 + 1 / self.n_classes)
                head.C = -lateral_inhib
                head.forward(emb - proto_mu)
                if head.li >= 500:
                    break

            # === Calculate meta-loss ===
            weights = head.output_weights
            test_ind = np.random.choice(len(validation_dataset), 64)
            test_set = [validation_dataset[ind] for ind in test_ind]

            accuracy = 0.0
            loss = 0.0
            for img, label in test_set:
                img = img.to(self.device)
                local_label = label_map[int(label)]
                query_emb = self.net(img)
                output = torch.tanh(query_emb @ weights.T)
                loss += F.cross_entropy(output, torch.tensor([local_label]).to(self.device)).item() / len(test_set)
                accuracy += (torch.argmax(output, dim=1) == local_label).float() / len(test_set)
        return loss, accuracy

