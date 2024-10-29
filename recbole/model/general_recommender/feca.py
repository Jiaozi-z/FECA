import numpy as np
import scipy.sparse as sp
import torch
from collections import OrderedDict
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType,build_sim,build_knn_normalized_graph

import torch.nn.functional as F

import os

class FECA(GeneralRecommender):
    r"""FECA is a GCN-based recommender model.
    FECA enhances recommendation accuracy and diversity by integrating textual features of users and items
    with interaction information. The model learns user and item representations by linearly propagating
    embeddings on the user-item interaction graph and combines them with textual features to obtain the final
    embedding through weighted summation. FECA also incorporates edge embeddings and popularity features to
    further improve recommendation performance.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FECA, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        self.text_dim=config["text_dim"]

        self.inter_features = dataset.get_inter_feature().to(self.device)
        self.INTER_TEXT_FIELD = config["INTER_TEXT_FIELD"]
        self.query_embeddings=self.inter_features[config["INTER_TEXT_FIELD"]]
        self.query_linear = torch.nn.Linear(self.text_dim, self.latent_dim) 
        self.user_item_query_map = self.get_user_item_query_map() 

        self.item_features = dataset.get_item_feature().to(self.device)
        self.item_price = self.item_features[config["ITEM_PRICE_FIELD"]].float()
        self.item_time = self.item_features[config["ITEM_TIME_FIELD"]].float()

        self.price_control_baseline = config['price_control_baseline']
        self.time_control_baseline = config['time_control_baseline']

        self.alignment_loss_weight = config['alignment_loss_weight']
        self.exposure_penalty_weight = config['exposure_penalty_weight']

        self.enable_edge_embedding = config['enable_edge_embedding']

        self.item_text_embeddings_origin = self.item_features[config["ITEM_TEXT_FIELD"]]
        self.item_text_embeddings = torch.nn.Embedding.from_pretrained(self.item_text_embeddings_origin, freeze=False)
        self.item_text_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_text_embeddings.weight.data)


        self.w_spe_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.rank_gate_network = torch.nn.Sequential(
            torch.nn.Linear(1, self.latent_dim),
            torch.nn.Sigmoid()
        )

    
        self.text_trs = torch.nn.Linear(self.item_text_embeddings_origin.shape[1], self.latent_dim)
        self.gate_t = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Sigmoid()
        )
        self.query_common = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.latent_dim, 1, bias=False)
        )

        self.gate_text_prefer = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Sigmoid()
        )

        self.text_transform = torch.nn.Linear(self.latent_dim, self.latent_dim)


        self.softmax = torch.nn.Softmax(dim=-1)
        self.sparse = config["sparse"]
        self.cl_loss_weight  = config[
            "cl_loss_weight"
        ] 
        self.knn_k = config["knn_k"]
        self.temperature = config["temperature"]
        self.text_original_adj = self.build_text_original_adj()
        
        self.item_popularity = self.calculate_item_popularity()

        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Sigmoid()
        )

        self.item_representation_network = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU()
        )
        self.popularity_transform = torch.nn.Linear(1, self.latent_dim)

        self.popularity_network = torch.nn.Sequential(
            torch.nn.Linear(1, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.attribute_network = torch.nn.Sequential(
            torch.nn.Linear(self.item_text_embeddings_origin.shape[1], self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.item_network = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.item_text_embeddings_origin.shape[1], self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.alignment_loss = torch.nn.MSELoss()

        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map
        self.best_user_embeddings = None
        self.best_item_embeddings = None

        

    def get_original_user_id(self, remapped_id):
        return self.user_id_map.get(remapped_id.item(), remapped_id.item())

    def get_original_item_id(self, remapped_id):
        return self.item_id_map.get(remapped_id.item(), remapped_id.item())
 
    def build_text_original_adj(self):
        text_adj = build_sim(self.item_text_embeddings.weight.detach())
        text_adj = build_knn_normalized_graph(
            text_adj, 
            topk=self.knn_k,         
            is_sparse=self.sparse,    
            norm_type='sym'         
        )
        return text_adj

    def calculate_item_popularity(self):
        # 计算每个商品的流行度特征
        item_ids = self.inter_features[self.ITEM_ID].cpu().numpy()
        unique, counts = np.unique(item_ids, return_counts=True)
        exposure_count = np.zeros(len(self.item_features[self.ITEM_ID]))
        exposure_count[unique] = counts
        return torch.tensor(exposure_count, dtype=torch.float32).to(self.device)


    def get_user_item_query_map(self):
        user_item_query_map = {}
        user_item_query_count = {}
        
        for user, item, query in zip(self.inter_features[self.USER_ID], 
                                     self.inter_features[self.ITEM_ID], 
                                     self.inter_features[self.INTER_TEXT_FIELD]):
            user_item_pair = (user.item(), item.item())
            if user_item_pair not in user_item_query_map:
                user_item_query_map[user_item_pair] = query
                user_item_query_count[user_item_pair] = 1
            else:
                user_item_query_map[user_item_pair] += query
                user_item_query_count[user_item_pair] += 1

        for user_item_pair in user_item_query_map:
            user_item_query_map[user_item_pair] /= user_item_query_count[user_item_pair]

        edge_embeddings = torch.zeros(self.interaction_matrix.nnz, self.text_dim, device=self.device)
        for idx, (user, item) in enumerate(zip(self.interaction_matrix.row, self.interaction_matrix.col)):
            user_item_pair = (user, item)
            if user_item_pair in user_item_query_map:
                edge_embeddings[idx] = user_item_query_map[user_item_pair]


        return user_item_query_map

    def get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        self.R = L.tolil()[:self.n_users, self.n_users:]
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings,item_embeddings

    def forward(self, querys=None, user_indices=None, item_indices=None):
        user_embeddings, item_embeddings = self.get_ego_embeddings()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        popularity_embeddings = self.popularity_network(self.item_popularity.unsqueeze(1))
        
        item_embeddings_with_id = torch.cat([self.item_embedding.weight, self.item_text_embeddings.weight], dim=1)
        item_embeddings_with_id = self.item_network(item_embeddings_with_id)
        

        text_feats = self.text_trs(self.item_text_embeddings.weight)
        text_item_embeds = torch.multiply(item_embeddings, self.gate_t(text_feats))

        if querys is not None and self.enable_edge_embedding:
            edge_embeddings = self.query_linear(querys)
            n_interactions = self.interaction_matrix.nnz
            full_edge_embeddings = torch.zeros(n_interactions, self.latent_dim, device=self.device)
            
            if user_indices is None or item_indices is None:
                batch_size = edge_embeddings.shape[0]
                full_edge_embeddings[:batch_size] = edge_embeddings
            else:
                for i, (user, item) in enumerate(zip(user_indices, item_indices)):
                    interaction_index = np.where((self.interaction_matrix.row == user.item()) & 
                                                (self.interaction_matrix.col == item.item()))[0]
                    if len(interaction_index) > 0:
                        full_edge_embeddings[interaction_index] = edge_embeddings[i]

            for layer_idx in range(self.n_layers):
                new_embeddings = self.propagate(embeddings_list[-1], full_edge_embeddings)
                embeddings_list.append(new_embeddings)

        else:
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
                embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        content_embeds = lightgcn_all_embeddings

        att_common = self.query_common(text_embeds)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common.unsqueeze(dim=1) * text_embeds
        sep_text_embeds = text_embeds - common_embeds
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_text_embeds + common_embeds) / 2

        if self.training == False:
            self.final_user_embeddings = user_all_embeddings.detach().cpu()
            self.final_item_embeddings = item_all_embeddings.detach().cpu()
        
        if (user_indices is None or item_indices is None) and querys is not None:
            self.best_user_embeddings = user_all_embeddings
            self.best_item_embeddings = item_all_embeddings
            return user_all_embeddings, item_all_embeddings, side_embeds, content_embeds, popularity_embeddings, item_embeddings_with_id
        return user_all_embeddings, item_all_embeddings 

    def propagate(self, embeddings, edge_embeddings):
        aggregated_embeddings = torch.sparse.mm(self.norm_adj_matrix, embeddings)
        
        # 考虑边嵌入的影响
        edge_influence = self.compute_edge_influence(embeddings, edge_embeddings)
        
        alpha=0.0001
        dropout_rate = 0.1
        updated_embeddings = F.relu(aggregated_embeddings) + F.relu(edge_influence * alpha)
        return updated_embeddings

    def compute_edge_influence(self, embeddings, edge_embeddings):
        user_indices = self.interaction_matrix.row
        item_indices = self.interaction_matrix.col + self.n_users
        
        user_influence = torch.zeros_like(embeddings)
        item_influence = torch.zeros_like(embeddings)
        
        user_influence.index_add_(0, torch.tensor(user_indices, device=self.device), edge_embeddings)
        item_influence.index_add_(0, torch.tensor(item_indices, device=self.device), edge_embeddings)
        
        user_degrees = torch.bincount(torch.tensor(user_indices, device=self.device), minlength=self.n_users).float() + 1e-7
        item_degrees = torch.bincount(torch.tensor(self.interaction_matrix.col, device=self.device), minlength=self.n_items).float() + 1e-7
        
        user_influence[:self.n_users] /= user_degrees.unsqueeze(1)
        item_influence[self.n_users:] /= item_degrees.unsqueeze(1)
        
        return user_influence + item_influence

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        unique_pairs = OrderedDict()
        for idx, (u, i) in enumerate(zip(user, pos_item)):
            unique_pairs[(u.item(), i.item())] = idx

        batch_queries = []
        for user_item_pair in unique_pairs:
            if user_item_pair in self.user_item_query_map:
                batch_queries.append(self.user_item_query_map[user_item_pair])
            else:
                batch_queries.append(torch.zeros(self.text_dim, device=self.device))
        batch_queries = torch.stack(batch_queries)

        user_all_embeddings, item_all_embeddings, side_embeds, content_embeds, popularity_embeddings, item_embeddings_with_id = self.forward(batch_queries)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

 
        
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        side_embeds_items = side_embeds_items[pos_item].mean(dim=1) 
        side_embeds_users = side_embeds_users[user].mean(dim=1) 

        max_item_index = min(side_embeds_items.size(0), content_embeds_items.size(0)) - 1
        valid_pos_item = torch.clamp(pos_item, 0, max_item_index)

        max_user_index = min(side_embeds_users.size(0), content_embeds_user.size(0)) - 1
        valid_user = torch.clamp(user, 0, max_user_index)

        cl_loss = self.InfoNCE(side_embeds_items[valid_pos_item], content_embeds_items[valid_pos_item], self.temperature) + \
                self.InfoNCE(side_embeds_users[valid_user], content_embeds_user[valid_user], self.temperature)

        alignment_loss = self.alignment_loss(popularity_embeddings, item_embeddings_with_id)

        pos_item_exposure = self.item_popularity[pos_item]
        

        exposure_penalty = torch.log(pos_item_exposure + 1) 
        exposure_loss = torch.mean(exposure_penalty * pos_scores)
        

        loss = mf_loss + self.reg_weight * reg_loss + self.cl_loss_weight * cl_loss + \
            self.alignment_loss_weight * alignment_loss + self.exposure_penalty_weight * exposure_loss

        print(f"mf_loss={mf_loss}, self.reg_weight * reg_loss={self.reg_weight * reg_loss}, "
            f"self.cl_loss_weight * cl_loss={self.cl_loss_weight * cl_loss}, "
            f"alignment_loss={ self.alignment_loss_weight *alignment_loss}, exposure_loss={self.exposure_penalty_weight * exposure_loss}")
        print(f"loss={loss}")


        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_item_pair = (user.item(), item.item())
        if user_item_pair in self.user_item_query_map:
            query_embedding = self.user_item_query_map[user_item_pair].unsqueeze(0)
        else:
            query_embedding = torch.zeros(1, self.text_dim).to(self.device)

        user_all_embeddings, item_all_embeddings = self.forward(query_embedding, user, item)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def adjust_time_weights(self, average_time, time_control_baseline,diff_weight=2.247, time_scale=0.0861):
        time_diff_ratio = (average_time - time_control_baseline ) / average_time
        time_control_weight = torch.exp(1 + diff_weight*torch.tensor(time_diff_ratio)) * time_scale
        return time_control_weight
    
    def adjust_price_weights(self, average_price, price_control_baseline,diff_weight=3.70731, price_scale=0.001651):
        price_diff_ratio = (average_price - price_control_baseline ) / average_price
        price_control_weight = torch.exp(1 + diff_weight*torch.tensor(price_diff_ratio)) * price_scale
        return price_control_weight
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        if not hasattr(self, 'final_user_embeddings') or not hasattr(self, 'final_item_embeddings'):
            self.final_user_embeddings = self.restore_user_e.detach().cpu()
            self.final_item_embeddings = self.restore_item_e.detach().cpu()
        
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        item_prices = self.item_price.unsqueeze(0).expand(scores.shape[0], -1)
        item_times = self.item_time.unsqueeze(0).expand(scores.shape[0], -1)

        average_price = item_prices.mean().item()
        average_time = item_times.mean().item()

        if self.price_control_baseline>0:
            price_control_weight=self.adjust_price_weights(average_price,self.price_control_baseline)
        else:
            price_control_weight=0

        if self.time_control_baseline>0:
            time_control_weight=self.adjust_time_weights(average_time,self.time_control_baseline)
        else:
            time_control_weight=0


        price_diff = (item_prices - self.price_control_baseline) / self.price_control_baseline
        price_effect = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff)) * price_control_weight
        scores -= price_effect

        time_diff = (item_times - self.time_control_baseline) / self.time_control_baseline
        time_effect = torch.where(time_diff > 0, time_diff, torch.zeros_like(time_diff)) * time_control_weight
        scores -= time_effect

        return scores.view(-1)
    


    def get_query_embedding(self, user, item):
        user_item_pair = (user.item(), item.item())
        if user_item_pair in self.user_item_query_map:
            return self.user_item_query_map[user_item_pair].unsqueeze(0)
        else:
            return torch.zeros(1, self.text_dim).to(self.device)
        
    def get_user_item_embedding(self, user, item):
        user_emb = self.user_embedding.weight[user]
        item_emb = self.item_embedding.weight[item]
        return user_emb, item_emb

    
    # def save_final_embeddings(self, save_dir='./feca_saved_embeddings'):
    #     if hasattr(self, 'final_user_embeddings') and hasattr(self, 'final_item_embeddings'):
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, 'final_user_embeddings.npy'), self.final_user_embeddings.numpy())
    #         np.save(os.path.join(save_dir, 'final_item_embeddings.npy'), self.final_item_embeddings.numpy())
    #         print("final_embeddings have been saved to", save_dir)
    #     else:
    #         print("no valid final_embeddings to save")

    # def save_best_epoch_embeddings(self, save_dir='./feca_best_epoch_saved_embeddings'):
    #     if self.best_user_embeddings is not None and self.best_item_embeddings is not None:
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, 'best_user_embeddings.npy'), self.best_user_embeddings.detach().cpu().numpy())
    #         np.save(os.path.join(save_dir, 'best_item_embeddings.npy'), self.best_item_embeddings.detach().numpy())
    #         print("best_embeddings have been saved to", save_dir)
    #     else:
    #         print("no valid best_embeddings to save")