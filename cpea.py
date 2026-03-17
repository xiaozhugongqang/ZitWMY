import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from utils import _l2norm, centering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Attention_AMMD(nn.Module):
    def __init__(self, dim, n_way, k_shot, num_head, is_proj=True):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.is_proj = is_proj
        self.n_way = n_way
        self.k_shot = k_shot
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        if self.is_proj:
            self.proj = nn.Linear(self.dim, self.dim)
        self.scale = (self.dim // self.num_head) ** 0.5

    def forward(self, x):
        # x: b, n, c, h, w
        # meta_prompt: num_prompt, c
        b, n, c, h, w = x.size()
        x = x.flatten(-2).transpose(-1, -2).contiguous()  # b, n, h*w, c
        x = x.view(b * n, -1, c)
        qkv = self.qkv(x).reshape(b * n, -1, 3, self.num_head, c // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b * n, -1, c)
        if self.is_proj:
            out = self.proj(out)
        out = x + out  # b*n, -1, c
        out = out.view(b, n, -1, c)
        return out



class MAP(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(MAP, self).__init__()


    def forward(self, task,feat):
        n, p, d = feat.shape
        m = task.shape[1]

        feature = torch.reshape(feat, (n, 1, p, d)) #[75,1,196,384]
        task = torch.reshape(task, (n, m, p, 1)) #[75,5,196,1]
        # dot = feature*task+feature #[75,5,196,384]
        dot = feature * task  # [75,5,196,384]
        dot = torch.mean(dot, dim=1) #[75,196,384]
        return dot



class TaskSpecificRegionSelector(nn.Module):

    def forward(self, feature, key_channels):
        "feature: (B, C, H, W)"
        "key_channels: (1, M, C)"
        n, p, d = feature.shape
        m = key_channels.shape[1]
        # B, M, C, H, W
        feature = torch.reshape(feature, (n, 1, p, d)) #[25,1,196,384]
        key_channels = torch.reshape(key_channels, (1, m, 1, d)) #[1,5,1,384]

        # B N H W
        dot = feature * key_channels #[25,5,196,384]
        dot = torch.mean(dot, dim=3) #[25,5,196]
        dot = torch.sigmoid(dot)

        return dot




class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim,  1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# PreNorm for Transformer
class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# MLP for Transformer
class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0., layer_scale_init=-1):
        super().__init__()
        self.layer_scale_init = layer_scale_init

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        if self.layer_scale_init > 0:
            self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * self.layer_scale_init)
        else:
            self.layer_scale = None

    def forward(self, x):

        return self.net(x) + x if self.layer_scale is None else self.net(x) * self.layer_scale + x


# SelfAttention Block for Transformer with einops
class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0., temperature=1., layer_scale_init=-1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.temperature = temperature
        self.layer_scale_init = layer_scale_init

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        if self.layer_scale_init > 0:
            self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * self.layer_scale_init)
        else:
            self.layer_scale = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)/self.temperature

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out) + x if self.layer_scale is None else self.to_out(out).mul_(self.layer_scale) + x


# Transformer Block for Transformer
class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_expansion=4, dropout=0., temperature=1., layer_scale_init=-1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = dim * mlp_expansion
        self.dropout = dropout
        self.temperature = temperature
        self.layer_scale_init = layer_scale_init

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, temperature=temperature,
                                       layer_scale_init=layer_scale_init)),
                PreNorm(dim, MLP(dim, self.mlp_dim, dropout=dropout, layer_scale_init=layer_scale_init))
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x)
            x = mlp(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class RFM(nn.Module):
#
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class channel_weight(nn.Module):
    """

    """

    def __init__(self, num_query, dim):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(channel_weight, self).__init__()

        self.num_query = num_query

        self.dim = dim

    def initialise(self):
        self.weight = nn.Parameter(
            data=torch.ones([self.num_query, self.dim]),
            requires_grad=True)
        print("Init successfully!")
        # self.train_lr_beta = nn.Parameter(
        #                 data=torch.ones([self.num_layers, self.total_num_inner_loop_steps]) * self.init_weight_decay * self.init_learning_rate,
        #                 requires_grad=True)

        # self.train_lr_alpha = nn.Parameter(
        #                 data=torch.ones([self.num_layers, self.total_num_inner_loop_steps]) * self.init_learning_rate,
        #                 requires_grad=True)

class CPEA(nn.Module):
    def __init__(self, in_dim=384):
        super(CPEA, self).__init__()

        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim/4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)

        self.fc2 = Mlp(in_features=196**2,  hidden_features=256, out_features=1)

        self.transformer = Transformer(384, 2, 1, 4, 0.0, 1.0, -1)
        self.task_descriptor = nn.Parameter(torch.randn(1, 60, 384))
        self.task_specific_region_selector = TaskSpecificRegionSelector()
        self.map = MAP(5, 384)

        # self.regularizer = nn.Sequential(
        #     nn.Linear(25, 25),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(25, 1))
        # self.init_weight = channel_weight(num_query=5 * 15, dim=384)
        # self.init_weight.initialise()

        # self.feat_dim = 384
        # self.n_way = 5
        # self.k_shot = 1
        # self.num_head = 8
        # self.attention = Attention_AMMD(self.feat_dim, self.n_way, self.k_shot, self.num_head,
        #                            is_proj=False)
        #
        # self.q = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        # self.k = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        # self.alpha = 1.0
        # self.num_groups = 1
        #
        # self.rfm = RFM(384, 64, 1)

    # def compute_attn_weight(self, global_f, local_f, dim):
    #     # global_f: b, nq/ns, c
    #     # # local_f: b, ns/nq, nf, c
    #     # local_v = local_f
    #     global_f = self.q(global_f)  # [1,75,384]
    #     local_f = self.k(local_f)  # [1,5,980,384]
    #     global_f, local_f = centering(global_f, local_f)
    #     global_f = _l2norm(global_f, dim=-1)
    #     local_f = _l2norm(local_f, dim=-1)
    #
    #     ng = global_f.size(dim=1)
    #     nl = local_f.size(dim=1)
    #
    #     if dim == 1:
    #         # global_f ==> support
    #         global_f = global_f.unsqueeze(dim).expand(-1, nl, -1, -1).unsqueeze(-1)  # b, nq, ns, c,1
    #         local_f = local_f.unsqueeze(dim + 1).expand(-1, -1, ng, -1, -1)  # b, nq, ns, nf,c
    #
    #     elif dim == 2:
    #         # global_f ==> query
    #         # global_f = global_f.unsqueeze(dim).expand(-1, -1, nl, -1).unsqueeze(-1) # b, nq, ns, c, 1
    #         global_f = global_f.unsqueeze(dim)  # [1,75,1,384]
    #         global_f = global_f.expand(-1, -1, nl, -1)  # [1,75,5,384]
    #         global_f = global_f.unsqueeze(-1)  # [1,75,5,384,1]
    #         local_f = local_f.unsqueeze(dim - 1).expand(-1, ng, -1, -1, -1)  # b, nq, ns, nf,c [1,75,5,980,384]
    #
    #     if self.num_groups > 1:
    #         b, nq, ns, nf, c = local_f.size()
    #         global_f = global_f.view(b, nq, ns, self.num_groups, -1, 1)  # b, nq, ns, ng, c_, 1
    #         local_f = local_f.view(b, nq, ns, nf, self.num_groups, -1).permute(0, 1, 2, 4, 3,
    #                                                                            5)  # b, nq, ns, ng, nf, c_,
    #         attn_weight = local_f @ global_f  # b, nq, ns, ng, nf, 1
    #     else:
    #         attn_weight = local_f @ global_f  # b, nq, ns, nf, 1 #[1,75,5,980,1]
    #
    #     attn_weight = attn_weight / self.alpha
    #     attn_weight = F.softmax(attn_weight.squeeze(dim=-1), dim=-1).unsqueeze(-1)
    #     attn = attn_weight
    #
    #     return attn
    # def compute_beta_gamma(self, support_xf, query_xf):
    #
    #     b, ns, nf, c = support_xf.size()
    #     nq = query_xf.size(1)
    #     # if self.cfg.model.mmd.switch == "all_supports":
    #     support_xf = support_xf.view(b, self.n_way, -1, c)  # b, nway, k_shot * nf, c [1,5,980,384]
    #     support_xf_global = support_xf.mean(dim=-2)  # b, nway, c [1,5,384]
    #     query_xf_global = query_xf.mean(dim=-2)  # b, nq, c [1,75,384]
    #
    #     beta = self.compute_attn_weight(query_xf_global, support_xf, dim=2)  # [1,75,5,980,1]
    #     gamma = self.compute_attn_weight(support_xf_global, query_xf, dim=1)  # [1,75,5,196,1]
    #
    #     return beta, gamma

    def forward(self, feat_query, feat_shot, args):
        # query: Q x n x C
        # feat_shot: KS x n x C
        _, n, c = feat_query.size()



        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query  # Q x n x C
        feat_shot  = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot  # KS x n x C
        feat_query = self.fc_norm1(feat_query)
        feat_shot  = self.fc_norm1(feat_shot)

        query_class = feat_query[:, 0, :].unsqueeze(1)  # [75,1,384]
        # query_class = feat_query[:, 1:, :].mean(1).unsqueeze(1) # [75,1,384]
        query_image = feat_query[:, 1:, :]  # Q x L x C[75,196,384]




        support_class = feat_shot[:, 0, :].unsqueeze(1)  # KS x 1 x C[25,1,384]
        # support_class = feat_shot[:, 1:, :].reshape(args.shot, -1, n -1, c).mean(2).mean(1).unsqueeze(1).unsqueeze(1) #[5,1,1,384]
        support_image = feat_shot[:, 1:, :]  # KS x L x C[25,196,384]

        # feat_query = query_image.contiguous()
        # feat_shot = support_image.contiguous().reshape(args.shot, -1, n - 1, c)
        # feat_shot = feat_shot.mean(dim=0)






        # #################可视化
        # support_image1 = support_image.contiguous().reshape(args.shot, -1, n - 1, c)
        # support_image1 = support_image1.mean(dim=0)
        # support_image1 = support_image1.view(-1,384) #[980,384]
        # # feat_query = feat_query.view(-1, 384) #[14700,384]
        # # feat_query_cpu = feat_query.cpu().detach().numpy()  # [14700,384]
        # support_image_cpu = support_image1.cpu().detach().numpy()  # [980,384]
        # # X = np.concatenate([feat_query_cpu,feat_shot_cpu]) #[15680,384]
        # # feature_lable = torch.cat([target_support, target_query], dim=0).cpu().detach().numpy()
        # # label = np.concatenate([feature_lable,sampled_label]) #[775]
        # colors = ["#ED0307", "#ED08D3", "#620AED", "#0B38ED", "#08C7ED", "#07ED7D", "#BBED09", "#ED9409", "#ED360B",
        #           "#000000"]
        # tsne = TSNE(random_state=0, perplexity=30.0, early_exaggeration=20.0)
        # X_TSNE = tsne.fit_transform(support_image_cpu)
        #
        # plt.figure(figsize=(10, 10))
        # plt.xlim(X_TSNE[:, 0].min() - 1, X_TSNE[:, 0].max() + 2)
        # plt.ylim(X_TSNE[:, 1].min() - 1, X_TSNE[:, 1].max() + 2)
        # print(len(support_image_cpu))
        # for i in range(len(support_image_cpu)):
        #     if i < 196:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[0], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     elif i >= 196 and i < 392:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[1], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     elif i >= 392 and i < 588:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[2], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     elif i >= 588 and i < 784:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[3], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     else:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[4], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        # # plt.xlabel('(a)',fontsize=40)
        # # plt.ylabel('y')
        # # plt.title("support set and query set",fontsize=40,family='Times New Roman')
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(color='black', linestyle='-.', linewidth=2)
        # plt.show()



        feat_query = query_image + 2.0 * query_class  # Q x L x C[75,196,384]
        feat_shot = support_image + 2.0 * support_class  # KS x L x C[5,196,384] [25,196,384]



        #*************************
        feat_shot1 = rearrange(feat_shot, 'n p c -> 1 (n p) c') #[1,4900,384]
        x = torch.cat([self.task_descriptor, feat_shot1], dim=1)  # [1,4905,384]
        x = self.transformer(x)
        task_descriptor = x[:, :60, :]  # [1,5,384]

        task_specific_support_region = self.task_specific_region_selector(feat_shot, task_descriptor)  # [25,5,196]
        task_specific_query_region = self.task_specific_region_selector(feat_query, task_descriptor) #[75,5,196]

        task_att_qry = self.map(task_specific_query_region, feat_query)  # [75,196,384]
        task_att_spt = self.map(task_specific_support_region, feat_shot)  # [25,196,384]




        task_att_qry = F.normalize(task_att_qry, p=2, dim=2)
        feat_query = task_att_qry - torch.mean(task_att_qry, dim=2, keepdim=True) #[75,196,384]

        feat_shot = task_att_spt.contiguous().reshape(args.shot, -1, n -1, c)  # K x S x n x C
        # feat_shot = task_att_spt.contiguous().reshape(args.shot, -1, 144, c)  # K x S x n x C
        # feat_shot = task_att_spt.contiguous().reshape(1, -1, n - 1, c)  # K x S x n x C
        feat_shot = feat_shot.mean(dim=0)  # S x n x C
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True) #[5,196,384]


        #
        # #################可视化
        # feat_shot = feat_shot.view(-1,384) #[980,384]
        # # feat_query = feat_query.view(-1, 384) #[14700,384]
        # # feat_query_cpu = feat_query.cpu().detach().numpy()  # [14700,384]
        # feat_shot_cpu = feat_shot.cpu().detach().numpy()  # [980,384]
        # # X = np.concatenate([feat_query_cpu,feat_shot_cpu]) #[15680,384]
        # # feature_lable = torch.cat([target_support, target_query], dim=0).cpu().detach().numpy()
        # # label = np.concatenate([feature_lable,sampled_label]) #[775]
        # colors = ["#ED0307", "#ED08D3", "#620AED", "#0B38ED", "#08C7ED", "#07ED7D", "#BBED09", "#ED9409", "#ED360B",
        #           "#000000"]
        # tsne = TSNE(random_state=0, perplexity=30.0, early_exaggeration=20.0)
        # X_TSNE = tsne.fit_transform(feat_shot_cpu)
        #
        # plt.figure(figsize=(10, 10))
        # plt.xlim(X_TSNE[:, 0].min() - 1, X_TSNE[:, 0].max() + 2)
        # plt.ylim(X_TSNE[:, 1].min() - 1, X_TSNE[:, 1].max() + 2)
        # print(len(feat_shot_cpu))
        # for i in range(len(feat_shot_cpu)):
        #     if i < 196:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[0], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     if i >= 196 and i < 392:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[1], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     if i >= 392 and i < 588:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[2], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     if i >= 588 and i < 784:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[3], fontweight='heavy',
        #                  fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        #     else:
        #         plt.text(X_TSNE[i, 0], X_TSNE[i, 1], '.', color=colors[4], fontweight='heavy',
        #                      fontsize=30, alpha=1)  # fontweight字体粗细程度，fontsize字号
        # # plt.xlabel('(a)',fontsize=40)
        # # plt.ylabel('y')
        # # plt.title("support set and query set",fontsize=40,family='Times New Roman')
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(color='black', linestyle='-.', linewidth=2)
        # plt.show()






        # similarity measure
        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx]  # n x C [196,384]
            tmp_query = tmp_query.unsqueeze(0)  # 1 x n x C

            # #*************
            # feat_shot = feat_shot*output_f[i]
            # tmp_query = tmp_query*output_f[i]

            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2))  # S x L x L

            # tmp_query = tmp_query.expand(feat_shot.shape)
            # out = torch.sqrt(torch.sum((feat_shot - tmp_query) ** 2, dim=-1))

            out = out.flatten(1)  # S x L*L
            out = self.fc2(out.pow(2))  # S x 1
            out = out.transpose(0, 1)  # 1 x S
            results.append(out)


        return results, None
