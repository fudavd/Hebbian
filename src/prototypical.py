import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import select_class_set

class MetaProto(nn.Module):
    def __init__(self, network, n_classes, n_shots=10, n_query=15, input_size=(1,28,28), device=None):
        super().__init__()
        self.net = network
        self.device = device or network.device
        # compute feature dimension from one forward pass
        with torch.no_grad():
            dummy = torch.randn((1,)+input_size, device=self.device)
            feat = self.net(dummy)
        self.feature_dim = feat.flatten().shape[-1]
        self.n_classes = n_classes
        self.n_shots = n_shots
        self.n_query = n_query
        print(f'\n###########  {n_classes}-way:\t{n_shots}-shot\n{network.description}\n')
        self.to(self.device)

    def forward(self, train_dataset, val_dataset=None):
        # === Construct support and query sets ===
        (support_set, query_set), label_map = self.get_support_and_query(train_dataset)

        # Compute embeddings for few-shot support set (ProtoNet style, similar to your Hebbian setup)
        proto_embeddings = torch.empty(self.n_classes, self.feature_dim, requires_grad=False).to(self.device)

        for c in range(self.n_classes):
            embeddings = torch.empty(self.n_shots, self.feature_dim, requires_grad=False).to(self.device)
            for i_shot in range(self.n_shots):
                img, label = support_set[c * self.n_shots + i_shot]
                emb = self.net(img.to(self.device))
                embeddings[i_shot] = emb
            # Compute the prototype for this class
            proto_embeddings[label_map[label]] = torch.mean(embeddings, dim=0)

        proto_vecs = proto_embeddings  # directly use prototypes

        # === Compute query embeddings and distances ===
        imgs_q = torch.stack([img.to(self.device) for img, _ in query_set])
        labels_q = torch.tensor([label_map[label] for _, label in query_set], device=self.device)

        emb_q = self.net(imgs_q)  # shape [n_classes * n_query, feature_dim]

        # Compute squared Euclidean distances (as in the paper) between query features and prototypes
        dists = torch.cdist(emb_q, proto_vecs, p=2)  # [Q, C]
        logits = - (dists ** 2)

        # === Loss & accuracy ===
        loss = F.cross_entropy(logits, labels_q)
        accuracy = (torch.argmax(logits, dim=1) == labels_q).float().mean()

        return loss, accuracy

    def get_support_and_query(self, dataset):
        unique_labels = np.random.permutation(np.unique(dataset.targets))
        support_set, query_set = [], []
        for c in unique_labels:
            one_class = select_class_set(dataset, [c])
            indices = np.random.randint(0, len(one_class), self.n_shots + self.n_query)
            for i in range(self.n_shots):
                support_set.append(one_class[indices[i]])
            for j in range(self.n_query):
                query_set.append(one_class[indices[self.n_shots + j]])
        label_map = {int(c): i for i, c in enumerate(unique_labels)}
        return (support_set, query_set), label_map

    def validation(self, validation_dataset):
        with torch.no_grad():
            (support_set, query_set), label_map = self.get_support_and_query(validation_dataset)

            imgs_sup = torch.stack([img.to(self.device) for img, _ in support_set])
            labels_sup = torch.tensor([label_map[label] for _, label in support_set], device=self.device)
            emb_sup = self.net(imgs_sup)

            prototypes = []
            for c in range(self.n_classes):
                class_emb = emb_sup[labels_sup == c]
                proto = class_emb.mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)

            test_ind = np.random.choice(len(validation_dataset), 64)
            test_set = [validation_dataset[ind] for ind in test_ind]

            accuracy = 0.0
            loss = 0.0
            with torch.no_grad():
                for img, label in test_set:
                    img = img.to(self.device)
                    emb = self.net(img)
                    local_label = label_map[label]
                    dists = torch.cdist(emb, prototypes, p=2)
                    logits = - (dists ** 2)
                    loss += F.cross_entropy(logits, torch.tensor([local_label]).to(self.device)).item() / len(test_set)
                    accuracy += (torch.argmax(logits, dim=1) == local_label).float() / len(test_set)
        return loss, accuracy
