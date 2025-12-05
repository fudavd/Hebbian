import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import functional_call

from src.utils import select_class_set


class MetaMamlRe(nn.Module):
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
        Perform a MAML-style meta-learning forward pass.
        """
        # === Construct support and query sets ===
        (support_set, query_set), label_map = self.get_support_and_query(train_dataset)

        # === Fast weights and classification head ===
        fast_weights = OrderedDict(self.net.named_parameters())
        head = nn.Linear(self.feature_dim, self.n_classes).to(self.device)
        fast_head = OrderedDict(head.named_parameters())

        # === Fast adaptation ===
        for step in range(5):
            imgs = torch.stack([img.to(self.device) for img, _ in support_set])
            labels = torch.tensor([label for _, label in support_set]).to(self.device)
            labels = torch.tensor([label_map[l.item()] for l in labels]).to(self.device)

            # Forward with current fast weights
            features = functional_call(self.net, fast_weights, (imgs,))
            outputs = functional_call(head, fast_head, (features,))

            loss = F.cross_entropy(outputs, labels)

            # Compute gradients and update
            grads = torch.autograd.grad(loss,
                                        list(fast_weights.values()) + list(fast_head.values()),
                                        create_graph=True)

            n_body = len(fast_weights)
            grads_body, grads_head = grads[:n_body], grads[n_body:]

            # Update full network
            fast_weights = OrderedDict(
                (name, param - self.lr_inner * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads_body)
            )
            fast_head = OrderedDict(
                (name, param - self.lr_inner * grad)
                for ((name, param), grad) in zip(fast_head.items(), grads_head)
            )
        support_loss = loss.item()
        support_accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean()

        # === Calculate meta-loss ===
        imgs = torch.stack([img.to(self.device) for img, _ in query_set])
        label = torch.tensor([label for _, label in query_set]).to(self.device)
        label = torch.tensor([label_map[l.item()] for l in label]).to(self.device)

        features = functional_call(self.net, fast_weights, (imgs,))
        outputs = functional_call(head, fast_head, (features,))
        query_loss = F.cross_entropy(outputs, label)
        query_accuracy = (torch.argmax(outputs, dim=1) == label).float().mean()

        # validation_loss, validation_accuracy = self.validation(val_dataset)
        return query_loss, query_accuracy

    def get_support_and_query(self, dataset):
        unique_labels = np.unique(dataset.targets)
        unique_labels = np.random.permutation(unique_labels)
        support_set, query_set = [], []
        for c in unique_labels:
            one_class_set = select_class_set(dataset, [c])
            indices = np.random.randint(0, len(one_class_set), self.n_shots + self.n_query)
            for i_shot in range(self.n_shots):
                img, label = one_class_set[indices[i_shot]]
                support_set.append((img, label))
            for i_meta in range(self.n_query):
                query_set.append(one_class_set[indices[self.n_shots + i_meta]])
        label_map = {c.item(): i for i, c in enumerate(unique_labels)}
        return (support_set, query_set), label_map


    def validation(self, validation_dataset):
        """
        validation meta-learning forward pass.
        validate on the validation set.
        """
        (support_set, _), label_map = self.get_support_and_query(validation_dataset)
        # === Fast weights and classification head ===
        fast_weights = OrderedDict(self.net.named_parameters())
        head = nn.Linear(self.feature_dim, self.n_classes).to(self.device)
        fast_head = OrderedDict(head.named_parameters())

        # === Fast adaptation ===
        for step in range(5):
            imgs = torch.stack([img.to(self.device) for img, _ in support_set])
            labels = torch.tensor([label for _, label in support_set]).to(self.device)
            labels = torch.tensor([label_map[l.item()] for l in labels]).to(self.device)

            # Forward with current fast weights
            features = functional_call(self.net, fast_weights, (imgs,))
            outputs = functional_call(head, fast_head, (features,))

            loss = F.cross_entropy(outputs, labels)

            # Compute gradients and update
            grads = torch.autograd.grad(loss,
                                        list(fast_weights.values()) + list(fast_head.values()),
                                        create_graph=True)

            n_body = len(fast_weights)
            grads_body, grads_head = grads[:n_body], grads[n_body:]

            # Update full network
            fast_weights = OrderedDict(
                (name, param - self.lr_inner * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads_body)
            )
            fast_head = OrderedDict(
                (name, param - self.lr_inner * grad)
                for ((name, param), grad) in zip(fast_head.items(), grads_head)
            )

        test_ind = np.random.choice(len(validation_dataset), 64)
        test_set = [validation_dataset[ind] for ind in test_ind]

        accuracy = 0.0
        loss = 0.0
        with torch.no_grad():
            for img, label in test_set:
                img = img.to(self.device)
                local_label = label_map[label]
                output = functional_call(self.net, fast_weights, (img,))
                output = functional_call(head, fast_head, (output,))
                loss += F.cross_entropy(output, torch.tensor([local_label]).to(self.device)).item()/len(test_set)
                accuracy += (torch.argmax(output, dim=1) == local_label).float()/len(test_set)
        return loss, accuracy


