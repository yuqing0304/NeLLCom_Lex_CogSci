import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class Sender(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(Sender, self).__init__()
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.hidden(x)
        out = self.classifier(h)
        return out


class Receiver(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_classes):
        super(Receiver, self).__init__()
        self.emb_layer = nn.Embedding(num_classes, embed_dim)
        self.feat_layer = nn.Linear(feat_dim, embed_dim)
    
    def forward(self, x, sender_out):
        # 处理[list1, list2, list3]
        imgs = []
        for i in range(len(x)):
            x_h = self.feat_layer(x[i])
            imgs.append(x_h)
        imgs = torch.stack(imgs) # [batch_size, game_size, embed_dim]
        # 处理sender输出
        h_s = self.emb_layer(sender_out) # [batch_size, embed_dim]
        h_s = h_s.unsqueeze(-1) # [batch_size, embed_dim, 1]
        # 计算接收器输出
        h = torch.bmm(imgs, h_s).squeeze(-1)
        log_probs = F.log_softmax(h, dim=1)
        return log_probs




# class NewReceiver(nn.Module):
#     def __init__(self, feat_dim, embed_dim, num_classes, dropout=0.2):
#         super(NewReceiver, self).__init__()
#         self.emb_layer = nn.Embedding(num_classes, embed_dim)
#         self.feat_layer = nn.Sequential(
#                             nn.Linear(feat_dim, embed_dim, bias=False),
#                             nn.BatchNorm1d(embed_dim),
#                             nn.ReLU(),
#                             nn.Dropout(p=dropout))
    
#     def forward(self, sender_out, receiver_input, _aux_input=None):

#         imgs = []
#         for i in range(receiver_input.size(1)):
#             x_h = self.feat_layer(receiver_input[:, i, :])
#             imgs.append(x_h)
#         imgs = torch.stack(imgs, dim=1) # [batch_size, game_size, embed_dim]
#         h_s = self.emb_layer(sender_out) # [batch_size, embed_dim]

#         h_s = h_s.unsqueeze(-1) # [batch_size, embed_dim, 1]
#         h = torch.bmm(imgs, h_s).squeeze(-1)
#         log_probs = F.log_softmax(h, dim=1)
#         return log_probs

class NewReceiver(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.2):
        super(NewReceiver, self).__init__()
        self.emb_layer = nn.Embedding(num_classes, hidden_dim)
        self.feat_layer = nn.Sequential(
                            nn.Linear(embed_dim, hidden_dim, bias=False),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(p=dropout))
    
    def forward(self, sender_out, receiver_input, _aux_input=None):
        imgs = []
        for i in range(receiver_input.size(1)):
            x_h = self.feat_layer(receiver_input[:, i, :])
            imgs.append(x_h)
        imgs = torch.stack(imgs, dim=1) 
        h_s = self.emb_layer(sender_out)
        h_s = h_s.unsqueeze(-1) 
        h = torch.bmm(imgs, h_s).squeeze(-1)
        log_probs = F.log_softmax(h, dim=1)
        return log_probs

    def get_embeddings(self):
        """Return current embeddings as NumPy array"""
        return self.emb_layer.weight.detach().cpu().numpy()

    def save_embeddings(self, filename="embeddings.npy", out_dir="outputs_emb"):
        """Save embeddings to a .npy file"""
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
        np.save(filepath, self.get_embeddings())
        print(f"Saved embeddings to {filepath}")

    # def plot_embeddings(self, title="Embeddings", savepath=None):
    #     """Plot embeddings with PCA and save or show"""
    #     embeddings = self.get_embeddings()
    #     pca = PCA(n_components=2)
    #     emb_2d = pca.fit_transform(embeddings)

    #     plt.figure(figsize=(8, 8))
    #     plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c="blue", s=50)
    #     for i in range(len(embeddings)):
    #         plt.text(emb_2d[i, 0] + 0.02, emb_2d[i, 1] + 0.02, str(i), fontsize=12)
    #     plt.title(title)

    #     if savepath:
    #         os.makedirs(os.path.dirname(savepath), exist_ok=True)
    #         plt.savefig(savepath, dpi=300, bbox_inches="tight")
    #         print(f"Saved plot to {savepath}")
    #     else:
    #         plt.show()




class NewSender(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.2, if_context=True):
        super(NewSender, self).__init__()
        self.if_context = if_context
        self.hidden_dim = hidden_dim  # Store hidden_dim as an instance attribute

        # Define layers
        self.hidden1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.dist1_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.dist2_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.joint_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, sender_input, _aux_input=None):
        x = sender_input[:, 0, :]
        dist1 = sender_input[:, 1, :]
        dist2 = sender_input[:, 2, :]

        # Process main input
        h = self.hidden1(x)

        # Process context layers but zero out outputs if if_context is False
        if not self.if_context:
            h1 = torch.zeros((dist1.size(0), self.hidden_dim), device=dist1.device)
            h2 = torch.zeros((dist2.size(0), self.hidden_dim), device=dist2.device)
        else:
            h1 = self.dist1_layer(dist1)
            h2 = self.dist2_layer(dist2)

        # Combine layers
        h_joint = torch.cat((h, h1, h2), dim=1)
        h = self.joint_layer(h_joint)

        # Classification layer
        out = self.classifier(h)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def set_context(self, if_context):
        """Update the context condition."""
        self.if_context = if_context


# class NewSender(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.2, if_context=True):
#         super(NewSender, self).__init__()
#         self.if_context = if_context
#         self.hidden_dim = hidden_dim  # Store hidden_dim as an instance attribute

#         # Define layers
#         self.hidden1 = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#         self.dist1_layer = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#         self.dist2_layer = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#         self.joint_layer = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def forward(self, sender_input, _aux_input=None):
#         x = sender_input[:, 0, :]
#         dist1 = sender_input[:, 1, :]
#         dist2 = sender_input[:, 2, :]

#         # Process main input
#         h = self.hidden1(x)

#         # Process context layers but zero out outputs if if_context is False
#         if not self.if_context:
#             h1 = torch.zeros((dist1.size(0), self.hidden_dim), device=dist1.device)
#             h2 = torch.zeros((dist2.size(0), self.hidden_dim), device=dist2.device)
#         else:
#             h1 = self.dist1_layer(dist1)
#             h2 = self.dist2_layer(dist2)

#         # Combine layers
#         h_joint = torch.cat((h, h1, h2), dim=1)
#         h = self.joint_layer(h_joint)

#         # Classification layer
#         out = self.classifier(h)
#         log_probs = F.log_softmax(out, dim=1)
#         return log_probs

#     def set_context(self, if_context):
#         """Update the context condition."""
#         self.if_context = if_context

