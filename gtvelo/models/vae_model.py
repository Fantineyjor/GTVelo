import torch as th
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

from torchdiffeq import odeint

from gtvelo.models.velocity_field import VelocityFieldReg
from gtvelo.models.modules import create_encoder, create_decoder, TransImg

from gtvelo.utils import normalize, sparse_mx_to_torch_sparse_tensor, batch_jacobian, gaussian_kl, paired_correlation, unique_index

from torch.distributions.normal import Normal
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
execution_countzzz = 0
class VAE(nn.Module):
 
    def __init__(self, observed = 2000, latent_dim = 20, zr_dim = 2, h_dim = 2,
                 encoder_hidden = 25, decoder_hidden = 25, 
                 root_weight = 0, num_steps = 100,
                 encoder_bn = False, decoder_bn = False, likelihood_model = 'gaussian',
                 include_time=False, kl_warmup_steps=25, kl_final_weight=1,
                  batch_correction=False, linear_decoder=True, linear_splicing=True, use_velo_genes=False,
                 correlation_reg = True, corr_weight_u = 0.1, corr_weight_s = 0.1, corr_weight_uu = 0., batches = 1, 
                 time_reg = False, time_reg_weight = 0.1, time_reg_decay=0, shared=False, corr_velo_mask=True, celltype_corr=False, celltypes=0, exp_time=False, max_sigma_z = 0,
                 latent_reg = False, velo_reg = False, velo_reg_weight = 0.0001, celltype_velo=False, gcn=True):
        super(VAE, self).__init__()
        
        # constant settings
        self.observed = observed
        self.latent = latent_dim
        self.zr_dim = zr_dim
        self.h_dim = h_dim
        self.root_weight = root_weight
        self.num_steps = num_steps
        self.gcn = gcn
        self.likelihood_model = likelihood_model
        self.include_time = include_time
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_final_weight = kl_final_weight
        self.batch_correction = batch_correction
        self.linear_decoder = linear_decoder
        self.linear_splicing = linear_splicing
        self.use_velo_genes = use_velo_genes
        self.correlation_reg = correlation_reg
        self.corr_weight_u = corr_weight_u
        self.corr_weight_s = corr_weight_s
        self.corr_weight_uu = corr_weight_uu
        self.batches = batches
        self.time_reg = time_reg
        self.time_reg_weight = time_reg_weight
        self.time_reg_decay = time_reg_decay
        self.shared = shared
        self.corr_velo_mask = corr_velo_mask
        self.celltype_corr = celltype_corr
        self.celltypes = celltypes
        self.exp_time = exp_time
        self.max_sigma_z = max_sigma_z
        self.latent_reg = latent_reg
        self.velo_reg = velo_reg
        self.velo_reg_weight = velo_reg_weight
        self.celltype_velo = celltype_velo
        self.annot = False
        #批次ID输进去不用图卷积
        if not gcn:
            self.batch_correction = True
            batch_correction = True
        
        # encoder networks
        self.encoder_z0_s, self.encoder_z0_u, self.encoder_z_s, self.encoder_z_u = create_encoder(observed, latent_dim, encoder_hidden)
        
        # h and t encoders

        self.encoder_c = TransImg(self.latent, 2*self.h_dim)#这里要combine
        self.encoder_t = TransImg(self.latent, 2)#这里要combine

            
        # decoder networks
        self.decoder_s, self.decoder_u = create_decoder(latent_dim, observed, decoder_hidden)
        
        # velocity field network；include_time是将时间包含在潜在动力学中
        self.velocity_field = VelocityFieldReg(self.latent, self.h_dim, self.zr_dim, include_time, linear_splicing = linear_splicing)
        
        # learnable decoder variance,代表了观测数据的条件方差，即模型预测的重构数据（例如，基因表达水平）的条件方差；潜在表示的条件方差，即模型中潜在变量的条件方差
        self.theta = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.observed) - 1/np.sqrt(self.observed))
        self.theta_z = nn.Parameter(2/np.sqrt(self.latent) * th.rand(self.latent) - 1/np.sqrt(self.latent))

        # regularize with linear splicing dynamics
        if self.velo_reg:#是否启用速度场正则化
            if self.celltype_velo:#是否考虑细胞类型对速度场正则化的影响
                self.beta = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.celltypes, self.observed) - 1/np.sqrt(self.observed))
                self.gamma = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.celltypes, self.observed) - 1/np.sqrt(self.observed))
            else:
                self.beta = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.observed) - 1/np.sqrt(self.observed))
                self.gamma = nn.Parameter(2/np.sqrt(self.observed) * th.rand(self.observed) - 1/np.sqrt(self.observed))
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        # initial state
        self.initial = nn.Linear(1, 2*self.latent).to(device)

    def decoder_x(self, x):
        return self.decoder_s[0](x)
                    
    def _run_dynamics(self, c, times, test=False):

        h0 = self.initial(th.zeros(c.shape[0], 1).cuda())
        h0 = th.cat((h0, th.zeros(c.shape[0], self.zr_dim).cuda(), c), dim=-1)
        if test:
            ht_full = odeint(self.velocity_field, h0, th.cat((th.zeros(1).cuda(), times), dim=-1), method='dopri8', options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        else:
            ht_full = odeint(self.velocity_field, h0, th.cat((th.zeros(1).cuda(), times), dim=-1), method='dopri5',rtol=1e-5, atol=1e-5, options=dict(max_num_steps=self.num_steps)).permute(1,0,2) #
        ht_full = ht_full[:,1:]
        ht = ht_full[...,:2*self.latent+self.zr_dim]

        return ht, h0#潜在状态ht和初始状态h0。
    
    def loss(self, normed_s, s, s_size_factor, mask_s, normed_u, u, u_size_factor, mask_u, velo_genes_mask, adj, root_cells, batch_id = (None, None, None, None), epoch = None):

        batch_id, batch_onehot, celltype_id, exp_time = batch_id
 
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, adj, batch_id = batch_onehot)
         
         
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]

        orig_index = th.arange(normed_s.shape[0]).cuda()

        velo_genes_mask = velo_genes_mask[0]
        sort_index, index = unique_index(latent_time)
        index = index.cuda()
        sort_index=sort_index.cuda()
        mask_s = mask_s[sort_index][index]
        mask_u = mask_u[sort_index][index]
        u = u[sort_index][index]

        s = s[sort_index][index]
        c=c[sort_index][index]
        root_cells = root_cells[sort_index][index]
        orig_index = orig_index[sort_index][index]
        s_size_factor = s_size_factor[sort_index][index]
        u_size_factor = u_size_factor[sort_index][index]
        latent_state = latent_state[sort_index][index]
        latent_mean = latent_mean[sort_index][index]
        latent_logvar = latent_logvar[sort_index][index]
        time_mean = time_mean[sort_index][index]
        time_logvar = time_logvar[sort_index][index]
        latent_time = latent_time[sort_index][index]
        z = z[sort_index][index]
        normed_s = normed_s[sort_index][index]
        
        normed_u = normed_u[sort_index][index]
        mask_s = th.mean(normed_s[mask_s>0])*mask_s
        mask_u = th.mean(normed_u[mask_u>0])*mask_u

        adj = adj.coalesce()
        min_length = min(len(sort_index), len(index))
        sort_index = sort_index[:min_length]
        index = index[:min_length]
        filtered_adj = adj.index_select(0, sort_index.cuda())
        filtered_adj = filtered_adj.index_select(1, index.cuda())
        adj = filtered_adj
 
        if self.batch_correction:
            batch_id = batch_id[sort_index][index]
            batch_onehot = batch_onehot[sort_index][index]
        if self.celltype_corr or self.celltype_velo:
            celltype_id = celltype_id[sort_index][index].long()
        if self.exp_time:
            exp_time = exp_time[sort_index][index]

        if not self.use_velo_genes:
            velo_genes_mask_ = th.ones_like(velo_genes_mask)
        else:
            velo_genes_mask_ = velo_genes_mask

        if not self.corr_velo_mask:
            velo_genes_mask = th.ones_like(velo_genes_mask)

        # run dynamics
        ht, h0 = self._run_dynamics(c, latent_time)
        zs, zu, zt = ht[np.arange(z.shape[0]), np.arange(z.shape[0]), :self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), self.latent:2*self.latent], ht[np.arange(z.shape[0]), np.arange(z.shape[0]), 2*self.latent:2*self.latent+self.zr_dim]

        zs_data, zu_data = z[...,:self.latent], z[...,self.latent:2*self.latent]

        shat_data = self.decoder_s(zs_data,adj)
        uhat_data = self.decoder_u(zu_data,adj)
        shat = self.decoder_s(zs,adj)
        uhat = self.decoder_u(zu,adj)

        scale = 1e-4 + F.softplus(self.theta)

        likelihood_dist_s_z = Normal(shat_data, scale)
        likelihood_dist_u_z = Normal(uhat_data,scale)

        likelihood_dist_s_latentz = Normal(shat, 1e-4 + F.softplus(self.theta))
        likelihood_dist_u_latentz = Normal(uhat, 1e-4 + F.softplus(self.theta))

        class NormalizationNetwork(nn.Module):
            def __init__(self, input_dim):
                super(NormalizationNetwork, self).__init__()
                self.fc = nn.Linear(input_dim, input_dim)
            
            def forward(self, x):
                return self.fc(x.cpu()).cuda()

        norm_net = NormalizationNetwork(z.size(-1))
        z = norm_net(z)

        if self.max_sigma_z > 0:
            likelihood_dist_z = Normal(th.cat((zs, zu), dim=-1), 1e-4 + self.max_sigma_z * th.sigmoid(th.cat(2*[self.theta_z], dim=-1)))
        else:
            likelihood_dist_z = Normal(th.cat((zs, zu), dim=-1), 1e-4 + F.softplus(th.cat(2*[self.theta_z], dim=-1))) #

        if len(root_cells.shape) > 1:
            root_cells = root_cells.squeeze(-1)

        if self.time_reg_decay > 0 and epoch != None:
            reconstruction_loss = (-(mask_s*velo_genes_mask_*likelihood_dist_s_z.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_z.log_prob(normed_u)).sum(-1) - likelihood_dist_z.log_prob(z).sum(-1) -
                            (mask_s*velo_genes_mask_*likelihood_dist_s_latentz.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_latentz.log_prob(normed_u)).sum(-1)  +
                        (1 - min(1, epoch/self.time_reg_decay))*self.root_weight*(root_cells * ((latent_time)**2) ) )
        else:
            reconstruction_loss = (-(mask_s*velo_genes_mask_*likelihood_dist_s_z.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_z.log_prob(normed_u)).sum(-1) - likelihood_dist_z.log_prob(z).sum(-1) -
                            (mask_s*velo_genes_mask_*likelihood_dist_s_latentz.log_prob(normed_s)).sum(-1) - (mask_u*velo_genes_mask_*likelihood_dist_u_latentz.log_prob(normed_u)).sum(-1)  +
                            self.root_weight*(root_cells * ((latent_time)**2) ) )
        corr_reg, corr_reg_val = self.corr_reg_func(normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask,adj)
        latent_reg = th.zeros(1).cuda()
        velo_reg = th.zeros(1).cuda()
        time_reg_ = th.zeros(1).cuda()

        validation_ae = th.sum(mask_s*(shat_data - normed_s)**2, dim=-1) + th.sum(mask_u*(uhat_data - normed_u)**2, dim=-1)
        validation_traj = th.sum(mask_s*(shat - normed_s)**2, dim=-1) + th.sum(mask_u*(uhat - normed_u)**2, dim=-1)

        if epoch != None: 
            kl_reg = self.kl_final_weight * min(1, epoch/self.kl_warmup_steps) * (gaussian_kl(latent_mean[:,:self.latent*2], latent_logvar[:,:self.latent*2]) + 0.1*gaussian_kl(time_mean[:,None], time_logvar[:,None]))
        else:
            kl_reg = self.kl_final_weight* (gaussian_kl(latent_mean[:,:self.latent*2], latent_logvar[:,:self.latent*2]) + 0.1*gaussian_kl(time_mean[:,None], time_logvar[:,None]))

        return reconstruction_loss + kl_reg + corr_reg + time_reg_ + latent_reg + velo_reg, validation_ae, validation_traj, corr_reg_val.unsqueeze(-1) + velo_reg.unsqueeze(-1), orig_index

    def latent_embedding(self, normed_s, normed_u, adj, batch_id=None):
        
        zs = self.encoder_z_s(normed_s, adj)
        zu = self.encoder_z_u(normed_u, adj)
        t_params = self.encoder_t(th.cat((zs, zu), dim=-1), adj)

        t_mean, t_logvar = t_params[:, 0], t_params[:, 1]
        latent_time = th.sigmoid(t_mean)  # 不使用采样

        context_params = self.encoder_c(th.cat((zs, zu), dim=-1), adj)

        context = context_params[:, :self.h_dim]  # 不使用采样
        
        z = th.cat((zs, zu), dim=-1)
        
        return th.cat((z, context), dim=-1), zs, zu, latent_time, t_mean, t_logvar 
    def batch_func(self, func, inputs, num_outputs, split_size = 500):

        outputs = [[] for j in range(num_outputs)]

        for i in range(split_size, inputs[0].shape[0] + split_size, split_size):

            inputs_i = []
            for input in inputs:
                if input==None or type(input) == int or type(input) == float or len(input.shape) == 1:
                    inputs_i.append(input)
                elif input.shape[0] != input.shape[1]:
                    inputs_i.append(input[i-split_size:i])
                else:
                    inputs_i.append(sparse_mx_to_torch_sparse_tensor(normalize(input[i-split_size:i, i-split_size:i])).cuda())

            outputs_i = func(*inputs_i)
            if type(outputs_i) != tuple:
                outputs_i = tuple((outputs_i,))

            if len(outputs_i) != num_outputs:
                print('error, expected different number of outputs')

            for j in range(num_outputs):
                outputs[j].append(outputs_i[j].cpu())
                
            outputs_tensor = [None for j in range(num_outputs)]
            for j in range(num_outputs):
                outputs_tensor[j] = th.cat(outputs[j], dim=0)

        return tuple(outputs_tensor)
    def reconstruct_latent(self, normed_s, normed_u, adj, batch_id = None):
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, normed_u, adj, batch_id = batch_id)
        
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]
        
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        
        # run dynamics
        ht, h0 = self._run_dynamics(c, unique_times, test=False)

        zs, zu, zt = ht[np.arange(z.shape[0]), inverse_indices,:self.latent], ht[np.arange(z.shape[0]), inverse_indices, self.latent:2*self.latent], ht[np.arange(z.shape[0]), inverse_indices, 2*self.latent:2*self.latent+self.zr_dim]
        
        velocity = self.velocity_field.drift(th.cat((z[:,:2*self.latent], zt, c, latent_time[:, None]), dim=-1))
        return th.cat((z,zt), dim=-1), th.cat((zs, zu, zt), dim=-1), velocity, latent_time, c
    


    def cell_trajectories(self, normed_s, normed_u, adj, batch_id = (None, None, None), mode='normal', time_steps = 50):

        batch_id, batch_onehot, celltype_id = batch_id

        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(normed_s, 
                                                                                                           normed_u, 
                                                                                                               adj, batch_id)
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]

        # choose times
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        times = th.linspace(0, unique_times.max(), time_steps).cuda()

        # run dyanmics
        ht, h0 = self._run_dynamics(c, times[1:], test=False)
        zs, zu, zt = ht[...,:self.latent], ht[..., self.latent:2*self.latent], ht[..., 2*self.latent:2*self.latent+self.zr_dim]
        z_traj = th.cat((zs, zu, zt), dim=-1)

        # decode
        if batch_id == None:
            x0 = th.cat((self.decoder_s(h0[:,:self.latent]), self.decoder_u(h0[:,self.latent:2*self.latent])), dim=-1)
        else:
            x0 = th.cat((self.decoder_batch(h0[:,:self.latent], batch_id), self.decoder_batch(h0[:,self.latent:2*self.latent], batch_id, 'u')), dim=-1)
            
        return th.cat((h0[:,None,:2*self.latent+self.zr_dim], z_traj), dim=1), times[None,:,None].repeat(z_traj.shape[0], 1, 1)

    def gene_trajectories(self, normed_s, normed_u, adj, gene_names, adata, exp_time=None, batch_id=(None, None, None), mode='normal', time_steps=50):
 
        batch_id, batch_onehot, celltype_id = batch_id

        gene_indices = [list(adata.var_names).index(gene) for gene in gene_names]

        gene_specific_s = normed_s[:, gene_indices]
        gene_specific_u = normed_u[:, gene_indices]

        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(
            gene_specific_s, gene_specific_u, adj, batch_id)
        
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]

        if exp_time is not None:
            # Normalize experimental time to match latent time scale
            exp_time_normalized = (exp_time - exp_time.min()) / (exp_time.max() - exp_time.min())
            exp_time_normalized = exp_time_normalized * latent_time.max()
            times = exp_time_normalized.cuda()
        else:
            # Use regular time steps
            unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
            times = th.linspace(0, unique_times.max(), time_steps).cuda()
        
        # Run dynamics for each gene
        gene_trajectories = []
        for gene_idx in range(len(gene_indices)):
            # Extract gene-specific components
            c_gene = c[gene_idx:gene_idx+1].repeat(time_steps-1, 1)
            
            # Run dynamics
            ht, h0 = self._run_dynamics(c_gene, times[1:], test=False)
            
            # Split into components
            zs, zu, zt = (ht[...,:self.latent], 
                        ht[...,self.latent:2*self.latent], 
                        ht[...,2*self.latent:2*self.latent+self.zr_dim])
            
            # Combine trajectories
            z_traj = th.cat((zs, zu, zt), dim=-1)
            
            # Add initial state
            full_traj = th.cat((h0[:,None,:2*self.latent+self.zr_dim], z_traj), dim=1)
            gene_trajectories.append(full_traj)
        
        # Stack all gene trajectories
        z_trajectories = th.stack(gene_trajectories)
        
        # Prepare time points matrix
        times_matrix = times[None,:,None].repeat(len(gene_indices), 1, 1)
        
        # Prepare additional gene-specific information
        gene_specific_data = {
            'gene_names': gene_names,
            'gene_indices': gene_indices,
            'latent_time': latent_time,
            'latent_mean': latent_mean,
            'latent_logvar': latent_logvar
        }
        
        return z_trajectories, times_matrix

    def gene_trajectories(self, normed_s, normed_u, adj, batch_id=(None, None, None), mode='normal', time_steps=50):
        """
        Internal method to compute gene trajectories
        """
        batch_id, batch_onehot, celltype_id = batch_id
        
        # Estimate latent state for genes
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar = self.latent_embedding(
            normed_s, normed_u, adj, batch_id)
        
        z = latent_state[:,:self.latent*2]
        c = latent_state[:,self.latent*2:]
        
        # Choose times
        unique_times, inverse_indices = th.unique(latent_time, return_inverse=True, sorted=True)
        times = th.linspace(0, unique_times.max(), time_steps).cuda()
        
        # Run dynamics
        ht, h0 = self._run_dynamics(c, times[1:], test=False)
        zs, zu, zt = (ht[...,:self.latent], 
                    ht[...,self.latent:2*self.latent], 
                    ht[...,2*self.latent:2*self.latent+self.zr_dim])
        z_traj = th.cat((zs, zu, zt), dim=-1)
        
        return th.cat((h0[:,None,:2*self.latent+self.zr_dim], z_traj), dim=1), times[None,:,None].repeat(z_traj.shape[0], 1, 1)
     

    def corr_reg_func(self, normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask,adj):

 
        gene_velocity = th.autograd.functional.jvp(lambda zs: self.decoder_s(zs, adj),zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
        gene_velocity_data = th.autograd.functional.jvp(lambda zs_data: self.decoder_s(zs_data, adj), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
        
    
        u_gene_velocity = th.autograd.functional.jvp(lambda zu: self.decoder_u(zu, adj), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
        u_gene_velocity_data = th.autograd.functional.jvp(lambda zu_data: self.decoder_u(zu_data, adj), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1] 
        m_s = (normed_s > 0)[:,velo_genes_mask==1] # normed_s
        m_u = (normed_u > 0)[:,velo_genes_mask==1] # normed_u
        
    
            
        corr_u = paired_correlation(gene_velocity[:,velo_genes_mask==1], uhat[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], uhat_data[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1], normed_u[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], normed_u[:,velo_genes_mask==1], m_u, dim=0)
        
        corr_s = paired_correlation(gene_velocity[:,velo_genes_mask==1], -shat[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], -shat_data[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity[:,velo_genes_mask==1], -normed_s[:,velo_genes_mask==1], m_s, dim=0) + paired_correlation(gene_velocity_data[:,velo_genes_mask==1], -normed_s[:,velo_genes_mask==1], m_s, dim=0)
        
        corr_uu = paired_correlation(u_gene_velocity[:,velo_genes_mask==1], -uhat[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], -uhat_data[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity[:,velo_genes_mask==1], -normed_u[:,velo_genes_mask==1], m_u, dim=0) + paired_correlation(u_gene_velocity_data[:,velo_genes_mask==1], -normed_u[:,velo_genes_mask==1], m_u, dim=0)
            
        corr_reg = -self.corr_weight_u * th.mean(corr_u) - self.corr_weight_s * th.mean(corr_s)  - self.corr_weight_uu * th.mean(corr_uu)
        corr_reg_val = -th.mean(corr_u) -th.mean(corr_s)  - th.mean(corr_uu)
        
        return corr_reg, corr_reg_val
    
    def latent_reg_func(self, zs, zu, zr, zs_data, zu_data, latent_time):
        
        u_splicing = []
        s_splicing = []
        s_splicing_data = []
        s_degradation = []
        split_size = 100
        for i in range(split_size, zs.shape[0] + split_size, split_size):
            
            if self.include_time:
                u_splicing.append(batch_jacobian(lambda x: self.velocity_field.unspliced_net(th.cat((x, zr[i-split_size:i], latent_time[i-split_size:i, None]), dim=-1)),
                                                 zu_data[i-split_size:i]).permute(1,0,2))
            else:
                u_splicing.append(batch_jacobian(lambda x: self.velocity_field.unspliced_net(th.cat((x, zr[i-split_size:i]), dim=-1)),
                                                 zu_data[i-split_size:i]).permute(1,0,2))
                
            s_splicing.append(batch_jacobian(lambda x: self.velocity_field.spliced_net(th.cat((zs[i-split_size:i], x), dim=-1)),
                                                 zu[i-split_size:i]).permute(1,0,2))

            s_splicing_data.append(batch_jacobian(lambda x: self.velocity_field.spliced_net(th.cat((zs_data[i-split_size:i], x), dim=-1)),
                                                 zu_data[i-split_size:i]).permute(1,0,2))
                
            s_degradation.append(batch_jacobian(lambda x: self.velocity_field.spliced_net(th.cat((x, zu_data[i-split_size:i]), dim=-1)),
                                                    zs_data[i-split_size:i]).permute(1,0,2))
        
        u_splicing = th.cat(u_splicing, dim=0)
        s_splicing = th.cat(s_splicing, dim=0)
        s_splicing_data = th.cat(s_splicing_data, dim=0)
        s_degradation = th.cat(s_degradation, dim=0)
        
        return 10000*(F.relu(-1*s_splicing).sum(dim=(-1,-2)) + F.relu(-1*s_splicing_data).sum(dim=(-1,-2)))#+ F.relu(u_splicing).sum(dim=(-1,-2)) + F.relu(s_degradation).sum(dim=(-1,-2)))



    def velo_reg_func(self, normed_s, normed_u, shat, uhat, shat_data, uhat_data, mask_s, mask_u, zs, zu, zt, zs_data, zu_data, latent_time, batch_id, celltype_id, velo_genes_mask):
        
        if self.batch_correction:
            
            if self.shared:
                gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                
                if self.include_time:
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    
                else:
                    
                    gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                    gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 's'), zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                    
                    u_gene_velocity = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(lambda x: self.decoder_batch(x, batch_id, 'u'), zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
                    
        else:
            if self.shared:
                gene_velocity = th.autograd.functional.jvp(self.decoder, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(self.decoder, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                if self.include_time:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                else:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder, zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
            else:
                gene_velocity = th.autograd.functional.jvp(self.decoder_s, zs, self.velocity_field.spliced_net(th.cat((zs, zu), dim=-1)), create_graph=True)[1]
                gene_velocity_data = th.autograd.functional.jvp(self.decoder_s, zs_data, self.velocity_field.spliced_net(th.cat((zs_data, zu_data), dim=-1)), create_graph=True)[1]
                
                if self.include_time:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt, latent_time[:,None]), dim=-1)), create_graph=True)[1]
                else:
                    u_gene_velocity = th.autograd.functional.jvp(self.decoder_u, zu, self.velocity_field.unspliced_net(th.cat((zu, zt), dim=-1)), create_graph=True)[1]
                    u_gene_velocity_data = th.autograd.functional.jvp(self.decoder_u, zu_data, self.velocity_field.unspliced_net(th.cat((zu_data, zt), dim=-1)), create_graph=True)[1]
        
        if self.celltype_velo:
            loss = []
            for i in range(self.celltypes):
                if th.sum(celltype_id == i) > 0:
                    splicing_velo_data = F.softplus(self.beta[i]) * normed_u[celltype_id == i] - F.softplus(self.gamma[i]) * normed_s[celltype_id == i]
                    splicing_velo = F.softplus(self.beta[i]) * uhat[celltype_id == i] - F.softplus(self.gamma[i]) * shat[celltype_id == i]
                    
                    loss.append( th.sum(th.sum((splicing_velo_data - gene_velocity_data[celltype_id == i])[:,velo_genes_mask==1]**2, dim=-1) + th.sum((splicing_velo - gene_velocity[celltype_id == i])[:,velo_genes_mask==1]**2, dim=-1), dim=0)  )
            loss = th.stack(loss).sum(0)/celltype_id.shape[0]
        else:
            splicing_velo_data = F.softplus(self.beta) * normed_u - F.softplus(self.gamma) * normed_s
            splicing_velo = F.softplus(self.beta) * uhat - F.softplus(self.gamma) * shat
            loss = th.sum((splicing_velo_data - gene_velocity_data)[:,velo_genes_mask==1]**2, dim=-1) + th.sum((splicing_velo - gene_velocity)[:,velo_genes_mask==1]**2, dim=-1)
        return loss
