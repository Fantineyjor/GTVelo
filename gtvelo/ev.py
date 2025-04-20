import numpy as np
import torch as th
import scanpy as sc
import scvelo as scv
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr, sem

from gtvelo.utils import paired_cosine_similarity
from sklearn.metrics import r2_score
from itertools import combinations

def true_velocity_cosine(adata, layer=True, estimated_key = 'velocity', true_key = 'true_velocity', plot=True):
    """
    Cosine similarity for true velocity in synthetic datasets
    """ 
    if layer:#是否从 adata.layers 中获取速度数据。如果设置为 False，则从 adata.obsm 中获取。
#检查数据里面是否存在NaN,如果有，则只使用非NaN，用np.isnan来找NaN的位置，并用逻辑索引来排除这些值
        if np.any(np.isnan(adata.layers[estimated_key])):
            cosine = paired_cosine_similarity(adata.layers[estimated_key][:,~np.isnan(adata.layers[estimated_key][0])], adata.layers[true_key][:,~np.isnan(adata.layers[estimated_key][0])])
        else:
            cosine = paired_cosine_similarity(adata.layers[estimated_key], adata.layers[true_key])
    else:
        if np.any(np.isnan(adata.obsm[estimated_key])):
            cosine = paired_cosine_similarity(adata.obsm[estimated_key][:,~np.isnan(adata.obsm[estimated_key][0])], adata.obsm[true_key][:,~np.isnan(adata.obsm[estimated_key][0])])
        else:
            cosine = paired_cosine_similarity(adata.obsm[estimated_key], adata.obsm[true_key])

    if plot:
        plt.boxplot(cosine)
        plt.show()
        
    return cosine, cosine.mean()
        
    
def true_velocity_correlation(adata, layer=True, estimated_key = 'velocity', true_key = 'true_velocity', plot=True, correlation = 'spearmanr'):
    """
    Correlation for true velocity in synthetic datasets
    """
    if correlation == 'spearmanr':
        func = spearmanr
    else:
        func = pearsonr
    
    if layer:
        corr = []
        for i in range(adata.layers[estimated_key].shape[1]):
            if not np.isnan(adata.layers[estimated_key][0,i]):
                corr.append(func(adata.layers[estimated_key][:,i], adata.layers[true_key][:,i])[0])
        corr = np.array(corr)
        print(corr.shape)
    else:
        corr = np.array([func(adata.obsm[estimated_key][:,i], adata.obsm[true_key][:,i])[0] for i in range(adata.obsm[estimated_key].shape[1])])

    if plot:
        plt.boxplot(corr)
        plt.show()
        
    return corr, corr.mean()

def velocity_magnitude_correlation(adata, layer=True, estimated_key = 'velocity', true_key = 'true_velocity', plot=True, correlation = 'spearmanr'):
    """
    Correlation for true velocity magnitudes
    """
    if layer:
        estimated = np.sqrt(np.nansum( (adata.obs['spliced_size_factor'].astype(float)[:,None] * np.array(adata.layers[estimated_key]) * adata.uns['scale_spliced'] )**2, axis=-1))
        true = np.sqrt(np.nansum(np.array(adata.layers[true_key])**2, axis=-1))

    else:
        estimated = np.sqrt(np.nansum(np.array(adata.obsm[estimated_key])**2, axis=-1))
        true = np.sqrt(np.nansum(np.array(adata.obsm[true_key])**2, axis=-1))
    
    return spearmanr(estimated, true)[0]
        
#计算单细胞数据集中潜在时间（latent time）和伪时间（pseudotime）之间的相关性
def pseudotime_correlation(adata, latent_time_key = 'latent_time', pseudotime_key = 'pseudotime', correlation = 'spearman', cell_label = None):

    if correlation == 'spearman':
        corr = spearmanr(adata.obs[latent_time_key], adata.obs[pseudotime_key])[0]
    elif correlation == 'pearson':
        corr = pearsonr(adata.obs[latent_time_key], adata.obs[pseudotime_key])[0]
    
    return corr


def nn_velo(adata, layer=True, batch_key = 'batch', vkey = 'velocity', neighbor_key = 'neighbors', xkey = None, n_neighbors = None, all=False, plot=False):
    
    if not all:
        
        index = np.random.choice(adata.shape[0], size = 100, replace=False)
    else:
        index = np.arange(adata.shape[0])

    values = []
    batch_ids = np.unique(adata.obs[batch_key])
    for i in range(len(index)):
        batch_i = [adata[adata.uns[neighbor_key]['indices'][index[i]][adata[adata.uns[neighbor_key]['indices'][index[i]]].obs[batch_key] == b]] for b in batch_ids]
        
        for (b1, b2) in list(combinations(np.arange(len(batch_ids)), 2)):
            
            if batch_i[b1].shape[0] > 0 and batch_i[b2].shape[0] > 0:
                if layer:
                    values.append(paired_cosine_similarity(batch_i[b1].layers[vkey].mean(0)[None], batch_i[b2].layers[vkey].mean(0)[None])[0])
                else:
                    values.append(paired_cosine_similarity(batch_i[b1].obsm[vkey].mean(0)[None], batch_i[b2].obsm[vkey].mean(0)[None])[0])
    
    values = np.array(values)
    
    if plot:
        plt.bar(np.arange(len(values)), values)
    
    return np.mean(values), sem(values), values
            
def keep_type(adata, nodes, target, k_cluster, scnt=False):
    """Select cells of targeted type
    Adapated from UniTVelo (Gao et al 2022).
    
    Args:
        adata (Anndata): 
            Anndata object.
        nodes (list): 
            Indexes for cells
        target (str): 
            Cluster name.
        k_cluster (str): 
            Cluster key in adata.obs dataframe
        scnt (bool):
            Whether to convert target to integer

    Returns:
        list: 
            Selected cells.
    """
    if scnt:
        target = int(target)
    return nodes[adata.obs[k_cluster][nodes].values == target]


#CBDir_1
def cross_boundary_correctness(
    adata, 
    cluster_key, 
    velocity_key, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap", majority_vote=False
):
    """Cross-Boundary Direction Correctness Score (A->B)
    Adapated from UniTVelo (Gao et al 2022).
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): 列表，包含元组 ("A", "B")，表示预期有从 A 群集到 B 群集的转移方向
            pairs of clusters has transition direction A->B
        return_raw (bool):  布尔值，指示是否返回原始分数，而不是聚合分数
            return aggregated or raw scores.
        x_emb (str): 键
            key to x embedding for visualization.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    
    x_emb = adata.obsm[x_emb] #adata.layers['spliced']#x_emb]
    # #######用scvelo进行速度降维后
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(velocity_key)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(velocity_key)][0]]
    ########用pca对layer里面的速度进行降维，此效果最好
    # v_emb = adata.layers[velocity_key]

    # from sklearn.decomposition import PCA

    # # 创建 PCA 对象，指定目标维度
    # pca = PCA(n_components=20)

    # # 拟合并转换 v_emb
    # v_emb = pca.fit_transform(v_emb)
    # x_emb = pca.fit_transform(x_emb)


    for u, v in cluster_edges:
        #以第一个循环为例
        #####scnt
        
        #u = int(u)#scnt，因为cluster_edge是数字，而u是str，所以要将u变成整数
        # sel = adata.obs[cluster_key] == u#选择细胞群0的细胞,scnt的细胞类型是数字，所以有问题，把引号去掉
        
        sel = adata.obs[cluster_key] == u#选择细胞群0的细胞,scnt的细胞类型是数字，所以有问题，把引号去掉
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]领域数设为0
        #print(f"u: {u}, sel sum: {sel.sum()}")
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, cluster_key), nbs)#筛选邻居（v15）中的节点
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue
            
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            if majority_vote:
                type_score.append(np.mean(dir_scores > 0) )
            else:
                type_score.append(np.mean(dir_scores))
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
         
    if return_raw:
        return all_scores #返回分化方向+此方向所有值
    
    return scores, np.mean([sc for sc in scores.values()])#返回 分化方向+此分化方向的均值+所有均值
    #########修改后的：
def cross_boundary_correctness2(
    adata, 
    cluster_key, 
    velocity_key, 
    cluster_edges,
    scnt=False,
    return_raw=False, 
    x_emb="X_umap", 
    majority_vote=False
):
    """Cross-Boundary Direction Correctness Score
    Returns all raw scores in a single column

    Args:
        adata (Anndata): 
            Anndata object.
        cluster_key (str):
            Key for clusters in adata.obs
        velocity_key (str):
            Key for velocity in adata.obsm
        cluster_edges (list):
            List of cluster edge pairs
        scnt (bool):
            Whether to convert cluster IDs to integers
        return_raw (bool):
            Whether to return raw scores
        x_emb (str):
            Key for embedding in adata.obsm
        majority_vote (bool):
            Whether to use majority voting

    Returns:
        Union[pd.DataFrame, Tuple[dict, float]]:
            Either raw scores DataFrame or (scores dict, mean score)
    """
    scores = {}
    all_scores = {}
    
    if x_emb == "X_umap": 
        v_emb = adata.obsm['{}_umap'.format(velocity_key)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(velocity_key)][0]]
    x_emb = adata.obsm[x_emb]

    for u, v in cluster_edges:
        if scnt:
            u = int(u)
             
        
        sel = adata.obs[cluster_key] == u
        nbs = adata.uns['neighbors']['indices'][sel]
        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, cluster_key, scnt), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue
            
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            if majority_vote:
                type_score.append(np.mean(dir_scores > 0))
            else:
                type_score.append(np.mean(dir_scores))
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    # Combine all scores into a single list
    all_values = []
    for scores_list in all_scores.values():
        all_values.extend(scores_list)

    # Create single-column DataFrame
    import pandas as pd
    df = pd.DataFrame(all_values, columns=['Scores'])
    
    if return_raw:
        return df
    
    return scores, np.mean([sc for sc in scores.values()])
#ICCo
def inner_cluster_coh(adata, cluster_key, velocity_key, return_raw=False, layer=True):
    """In-cluster Coherence Score.
    Adapated from UniTVelo (Gao et al 2022).
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        return_raw (bool): 
            return aggregated or raw scores.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[cluster_key])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[cluster_key] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, cluster_key), nbs)

        if layer:
            velocities = adata.layers[velocity_key]
        else:
            velocities = adata.obsm[velocity_key]
        
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
     
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])
import numpy as np
from scipy.stats import pearsonr, spearmanr

def inner_cluster_coh2(adata, cluster_key, velocity_key, layer=True, min_cells=5, metric='pearson', lambda_=1):
    """In-cluster Coherence Score (Improved).
    
    Args:
        adata (Anndata): Anndata object.
        cluster_key (str): Key to the cluster column in adata.obs DataFrame.
        velocity_key (str): Key to the velocity matrix in adata.obsm or adata.layers.
        layer (bool): Whether velocity matrix is in adata.layers (True) or adata.obsm (False).
        min_cells (int): Minimum number of cells required in a cluster to calculate score.
        metric (str): Metric to use for calculating similarity, 'pearson' or 'spearman'.
        lambda_ (float): Balance factor for variance penalty.
        
    Returns:
        dict: Mean scores indexed by cluster.
        float: Averaged score over all clusters.
    """
    clusters = adata.obs[cluster_key].unique()
    scores = {}
    
    if layer:
        velocities = adata.layers[velocity_key]
    else:
        velocities = adata.obsm[velocity_key]
    
    for cat in clusters:
        sel = adata.obs[cluster_key] == cat
        cat_vels = velocities[sel]
        
        if cat_vels.shape[0] >= min_cells:
            if metric == 'pearson':
                similarities = np.array([[pearsonr(x, y)[0] for y in cat_vels] for x in cat_vels])
            elif metric == 'spearman':
                similarities = np.array([[spearmanr(x, y)[0] for y in cat_vels] for x in cat_vels])
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            np.fill_diagonal(similarities, 0)  # 将对角线元素设为0,排除自身相似性
            
            # 计算每个细胞与其他细胞相似性的平均值和标准差
            mu = np.mean(similarities, axis=1)
            sigma = np.std(similarities, axis=1)
            
            # 计算幅度差异惩罚项
            amplitudes = np.linalg.norm(cat_vels, axis=1)
            amplitude_diff = np.abs(amplitudes[:, np.newaxis] - amplitudes)
            np.fill_diagonal(amplitude_diff, 0)  # 将对角线元素设为0,排除自身差异
            penalty = np.exp(-np.mean(amplitude_diff, axis=1))
            
            # 计算每个细胞的一致性得分
            cell_scores = penalty * (mu - lambda_ * sigma)
            
            scores[cat] = np.mean(cell_scores)
    
    cluster_scores = np.mean(list(scores.values())) if scores else 0.0
    
    return scores, cluster_scores

def integration_metrics(adata_raw, label_key, batch_key, emb_key, n_neighbors=30):

    import scib
    from scib.metrics.silhouette import silhouette_batch, silhouette
    from scib.metrics.lisi import lisi_graph

    adata = adata_raw.copy()

    adata.obsm['X_emb'] = adata.obsm[emb_key].copy()
    scv.pp.neighbors(adata, use_rep='X_emb', n_neighbors=n_neighbors)
    
    # biological conservation
    #isolated_labels = scib.metrics.isolated_labels(adata, label_key=label_key, batch_key=batch_key, embed='X_emb', verbose=False)
    #silhouette = scib.metrics.silhouette(adata, group_key=label_key, embed='X_emb')
    # add trajctory conseraion
    
    # batch correction
    kbet = scib.metrics.kBET(adata, label_key=label_key, batch_key=batch_key, embed='X_emb')
    #silhouette_batch = scib.metrics.silhouette_batch(adata, group_key=label_key, batch_key=batch_key, embed='X_emb', verbose=False)
    
    # add lisi
    ilisi, clisi = scib.metrics.lisi.lisi_graph(adata, batch_key=batch_key, label_key=label_key, type_='embed', n_cores=1)
    
    df = {
        'cLISI': [clisi],
        'kBET': [kbet],
        'iLISI': [ilisi]
    }

    #df = {
    #    'isolated label': [isolated_labels],
    #    'silhouette labels': [silhouette],
    #    'clisi': [clisi],
    #    'kBET': [kbet],
    #    'silhouette batch': [silhouette_batch],
    #    'ilisi': [ilisi]
    #}
    
    return df

from itertools import chain
def flatten_dict_values(dictionary):
    return list(chain(*dictionary.values()))

def create_real_dataframe(dicts, names):
    
    results = {}
    for i in range(len(dicts)):
        results[names[i]] = np.array(flatten_dict_values(dicts[i]))
    
    max_len = max([len(results[key]) for key in results.keys()]) 
    
    values = []
    keys = []
    for key in results.keys():
        if len(results[key]) < max_len:
            new_array = np.ones(max_len) * np.nan
            new_array[:len(results[key])] = results[key]
            results[key] = new_array
            values.append(new_array)
            keys.append([key]*max_len)
        else:
            values.append(results[key])
            keys.append([key]*max_len)
    values = pd.DataFrame({'scores': np.array(values).flatten(), 'Dataset': np.array(keys).flatten()})
    
    return values 
    

