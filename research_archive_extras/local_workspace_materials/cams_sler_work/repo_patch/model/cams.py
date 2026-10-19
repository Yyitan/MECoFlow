import math
import torch
from torch import nn
from clip_modules import clip
import torch.nn.functional as F
from model.csp import CSPInterface
from clip_modules.text_encoder import CustomTextEncoder

""""
    CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement 
"""

class CAMS(nn.Module):

    def __init__(self,args,dataset):
        super(CAMS, self).__init__()
        self.args = args
        self.dset = dataset
        
        self.clip_model = clip.load_clip(self.args) # Load the pre-trained CLIP. See "Learning Transferable Visual Models From Natural Language Supervision".
        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = CustomTextEncoder(self.clip_model)
        
        self.csp = CSPInterface(clip.tokenize, self.clip_model, self.dset,self.args) # Use the soft prompt method. See "Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning".
        attr2idx = self.dset.attr2idx
        obj2idx = self.dset.obj2idx
        self.train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                         for attr, obj in self.dset.train_pairs])        

        self.temp_logit = nn.Parameter(self.clip_model.logit_scale.exp().detach()) # Temperature
        self._affinity_diagnostics_logged = False
        if getattr(self.args, 'use_sler', False):
            embed_dim = self.clip_model.text_projection.shape[1]
            self.sler_composer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # Counterfactual composition starts from an interpretable average
            # of primitive evidence and learns only a residual correction.
            nn.init.zeros_(self.sler_composer[-1].weight)
            nn.init.zeros_(self.sler_composer[-1].bias)
            adapter_dim = getattr(self.args, 'sler_adapter_dim', 64)
            self.sler_query_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, adapter_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(adapter_dim, embed_dim, bias=False),
                )
                for _ in range(3)
            ])
            for adapter in self.sler_query_adapters:
                nn.init.zeros_(adapter[-1].weight)
            gate_init = min(max(getattr(self.args, 'sler_gate_init', 0.1), 1e-4), 1.0 - 1e-4)
            self.sler_gate_logits = nn.Parameter(
                torch.full((3,), math.log(gate_init / (1.0 - gate_init)))
            )

    def soft_cross_entropy(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()

    def grouped_logsumexp(self, pair_logits, group_ids, num_groups):
        """Marginalize pair logits into primitive logits in log space."""
        batch_size, num_pairs = pair_logits.shape
        expanded_groups = group_ids.unsqueeze(0).expand(batch_size, num_pairs)
        maxima = pair_logits.new_full((batch_size, num_groups), float('-inf'))
        if hasattr(maxima, 'scatter_reduce_'):
            maxima.scatter_reduce_(
                1, expanded_groups, pair_logits, reduce='amax', include_self=True
            )
            maxima = maxima.detach()
            stabilized = torch.exp(
                pair_logits - maxima.gather(1, expanded_groups)
            )
            sums = stabilized.new_zeros((batch_size, num_groups))
            sums.scatter_add_(1, expanded_groups, stabilized)
            return maxima + sums.clamp_min(1e-12).log()

        # Compatibility fallback for older PyTorch builds.
        return torch.stack([
            torch.logsumexp(pair_logits[:, group_ids == index], dim=1)
            for index in range(num_groups)
        ], dim=1)

    def structured_marginal_losses(self, att_branch, obj_branch, com_branch,
                                   glb_branch, att_id, obj_id):
        """Train pair scores and primitive branches as one coherent posterior."""
        train_pairs = self.train_pairs.to(com_branch.device)
        pair_logits = 0.5 * (com_branch + glb_branch)
        att_marginal = self.grouped_logsumexp(
            pair_logits, train_pairs[:, 0], len(self.dset.attrs)
        )
        obj_marginal = self.grouped_logsumexp(
            pair_logits, train_pairs[:, 1], len(self.dset.objs)
        )
        marginal_loss = 0.5 * (
            F.cross_entropy(att_marginal, att_id) +
            F.cross_entropy(obj_marginal, obj_id)
        )

        temperature = max(getattr(self.args, 'smc_temperature', 2.0), 1e-6)
        row_idx = torch.arange(att_id.shape[0], device=att_id.device)

        def reliable_distillation(student_logits, teacher_logits, truth):
            teacher_prob = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
            teacher_log_prob = teacher_prob.clamp_min(1e-12).log()
            student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
            per_sample = (
                teacher_prob * (teacher_log_prob - student_log_prob)
            ).sum(dim=-1) * (temperature ** 2)
            reliability = teacher_prob[row_idx, truth]
            return (per_sample * reliability).sum() / reliability.sum().clamp_min(1e-6)

        consistency_loss = 0.5 * (
            reliable_distillation(att_marginal, att_branch, att_id) +
            reliable_distillation(obj_marginal, obj_branch, obj_id)
        )
        return marginal_loss, consistency_loss

    def latent_evidence_logits(self, pooled_reps, unit_reps, text_reps, branch_idx,
                               return_routes=False):
        """Route every textual class to its own sparse latent visual evidence."""
        pooled_reps = F.normalize(pooled_reps, p=2, dim=-1)
        unit_reps = F.normalize(unit_reps, p=2, dim=-1)
        text_reps = F.normalize(text_reps, p=2, dim=-1)
        route_text_reps = text_reps
        stable_routing = getattr(self.args, 'sler_freeze_legacy', False)
        if stable_routing:
            route_text_reps = F.normalize(
                text_reps + self.sler_query_adapters[branch_idx](text_reps),
                p=2,
                dim=-1,
            )

        tau = max(getattr(self.args, 'sler_route_tau', 0.07), 1e-6)
        mix = getattr(self.args, 'sler_mix', 0.5)
        if stable_routing:
            mix = mix * torch.sigmoid(self.sler_gate_logits[branch_idx])
        branch_scales = getattr(self.args, 'sler_branch_scales', [1.0, 1.0, 1.0])
        mix = mix * float(branch_scales[branch_idx])
        confidence_floor = getattr(self.args, 'sler_confidence_floor', 0.25)
        chunk_size = getattr(self.args, 'sler_class_chunk_size', 512)
        if chunk_size is None or chunk_size <= 0:
            chunk_size = text_reps.shape[0]

        logits_chunks, route_chunks = [], []
        log_num_units = math.log(max(unit_reps.shape[1], 2))
        for start in range(0, text_reps.shape[0], chunk_size):
            text_chunk = text_reps[start:start + chunk_size]
            route_text_chunk = route_text_reps[start:start + chunk_size]
            pooled_cos = pooled_reps @ text_chunk.T
            unit_cos = torch.einsum('bkd,cd->bkc', unit_reps, route_text_chunk)
            routes = torch.softmax(unit_cos / tau, dim=1)
            evidence_cos = (routes * unit_cos).sum(dim=1)
            route_entropy = -(routes.clamp_min(1e-12) * routes.clamp_min(1e-12).log()).sum(dim=1)
            route_confidence = 1.0 - route_entropy / log_num_units
            adaptive_mix = mix * (
                confidence_floor + (1.0 - confidence_floor) * route_confidence
            )
            fused_cos = pooled_cos + adaptive_mix * (evidence_cos - pooled_cos)
            logits_chunks.append(self.temp_logit * fused_cos)
            if return_routes:
                route_chunks.append(routes)

        logits = torch.cat(logits_chunks, dim=-1)
        routes = torch.cat(route_chunks, dim=-1) if return_routes else None
        return logits, routes, unit_reps

    def compute_branch_logits(self, img, com_prompt_reps, att_prompt_reps, obj_prompt_reps,
                              return_routes=False):
        encoded = self.image_encoder(img)
        att, obj, com, glb = encoded[:4]

        if not getattr(self.args, 'use_sler', False):
            att_logits = self.temp_logit * F.normalize(att, p=2, dim=-1) @ att_prompt_reps.T
            obj_logits = self.temp_logit * F.normalize(obj, p=2, dim=-1) @ obj_prompt_reps.T
            com_logits = self.temp_logit * F.normalize(com, p=2, dim=-1) @ com_prompt_reps.T
            glb_logits = self.temp_logit * F.normalize(glb, p=2, dim=-1) @ com_prompt_reps.T
            return att_logits, obj_logits, com_logits, glb_logits, None

        att_units, obj_units, com_units = encoded[4:7]
        att_logits, att_routes, att_units = self.latent_evidence_logits(
            att, att_units, att_prompt_reps, branch_idx=0, return_routes=return_routes)
        obj_logits, obj_routes, obj_units = self.latent_evidence_logits(
            obj, obj_units, obj_prompt_reps, branch_idx=1, return_routes=return_routes)
        com_logits, com_routes, com_units = self.latent_evidence_logits(
            com, com_units, com_prompt_reps, branch_idx=2, return_routes=return_routes)
        glb_logits = self.temp_logit * F.normalize(glb, p=2, dim=-1) @ com_prompt_reps.T
        route_info = None
        if return_routes:
            route_info = {
                'att_routes': att_routes,
                'obj_routes': obj_routes,
                'com_routes': com_routes,
                'att_units': att_units,
                'obj_units': obj_units,
                'com_units': com_units,
            }
        return att_logits, obj_logits, com_logits, glb_logits, route_info

    def sler_auxiliary_losses(self, att_id, obj_id, route_info):
        row_idx = torch.arange(att_id.shape[0], device=att_id.device)
        att_routes = route_info['att_routes'].permute(0, 2, 1)[row_idx, att_id]
        obj_routes = route_info['obj_routes'].permute(0, 2, 1)[row_idx, obj_id]

        pair_lookup = self.train_pairs.to(att_id.device)
        # pair_id is not needed here: each training example's composition
        # route is gathered by matching its primitive ids.
        pair_match = (
            (pair_lookup[:, 0].unsqueeze(0) == att_id.unsqueeze(1)) &
            (pair_lookup[:, 1].unsqueeze(0) == obj_id.unsqueeze(1))
        )
        pair_id = pair_match.float().argmax(dim=1)
        com_routes = route_info['com_routes'].permute(0, 2, 1)[row_idx, pair_id]

        compositional_target = 0.5 * (att_routes.detach() + obj_routes.detach())
        route_consistency = -(
            compositional_target * com_routes.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        entropy_norm = math.log(max(att_routes.shape[1], 2))
        primitive_entropy = -0.5 * (
            (att_routes.clamp_min(1e-12) * att_routes.clamp_min(1e-12).log()).sum(dim=-1) +
            (obj_routes.clamp_min(1e-12) * obj_routes.clamp_min(1e-12).log()).sum(dim=-1)
        ).mean() / entropy_norm
        route_loss = route_consistency + getattr(self.args, 'sler_entropy_weight', 0.05) * primitive_entropy

        if att_id.shape[0] < 2:
            return route_loss, route_loss.new_zeros(())

        permutation = torch.roll(row_idx, shifts=1)
        att_evidence = (
            att_routes.unsqueeze(-1) * route_info['att_units']
        ).sum(dim=1)
        obj_evidence = (
            obj_routes.unsqueeze(-1) * route_info['obj_units']
        ).sum(dim=1)
        counterfactual_pairs = torch.stack([att_id, obj_id[permutation]], dim=1)
        counterfactual_input = torch.cat(
            [att_evidence, obj_evidence[permutation]], dim=-1
        )
        counterfactual_feature = F.normalize(
            0.5 * (att_evidence + obj_evidence[permutation]) +
            self.sler_composer(counterfactual_input),
            p=2,
            dim=-1,
        )
        token_com, _, _ = self.csp.construct_token_tensors(counterfactual_pairs)
        counterfactual_text = self.text_encoder(
            self.csp.token_ids[0], token_com, enable_pos_emb=True
        )
        counterfactual_text = F.normalize(counterfactual_text, p=2, dim=-1)
        counterfactual_logits = self.temp_logit * counterfactual_feature @ counterfactual_text.T
        same_pair = (counterfactual_pairs[:, None] == counterfactual_pairs[None, :]).all(dim=-1)
        counterfactual_targets = same_pair.to(counterfactual_logits.dtype)
        counterfactual_targets = counterfactual_targets / counterfactual_targets.sum(dim=-1, keepdim=True)
        counterfactual_loss = self.soft_cross_entropy(counterfactual_logits, counterfactual_targets)
        return route_loss, counterfactual_loss

    def build_pair_affinity_targets(self, pair_id, att_id, obj_id, com_prompt_reps):
        device = com_prompt_reps.device
        num_pairs = self.train_pairs.shape[0]
        eps = getattr(self.args, 'affinity_eps', 0.0)

        one_hot = F.one_hot(pair_id, num_classes=num_pairs).to(device=device, dtype=com_prompt_reps.dtype)
        if not getattr(self.args, 'use_pair_affinity', False) or eps <= 0:
            return one_hot

        train_pairs = self.train_pairs.to(device)
        attr_match = (train_pairs[:, 0].unsqueeze(0) == att_id.unsqueeze(1)).to(com_prompt_reps.dtype)
        obj_match = (train_pairs[:, 1].unsqueeze(0) == obj_id.unsqueeze(1)).to(com_prompt_reps.dtype)

        text_sim = (com_prompt_reps.detach() @ com_prompt_reps.detach().T).to(com_prompt_reps.dtype)
        text_sim = text_sim[pair_id]

        affinity = (
            self.args.affinity_attr_weight * attr_match +
            self.args.affinity_obj_weight * obj_match +
            self.args.affinity_text_weight * text_sim
        )
        tau = max(getattr(self.args, 'affinity_tau', 0.07), 1e-6)
        if getattr(self.args, 'affinity_exclude_gt', False):
            affinity = affinity.masked_fill(one_hot.bool(), float('-inf'))
        affinity_dist = torch.softmax(affinity / tau, dim=-1)
        targets = (1.0 - eps) * one_hot + eps * affinity_dist

        if not self._affinity_diagnostics_logged:
            with torch.no_grad():
                row_idx = torch.arange(pair_id.shape[0], device=device)
                gt_affinity_prob = affinity_dist[row_idx, pair_id]
                gt_target_prob = targets[row_idx, pair_id]
                non_gt_affinity = affinity_dist.masked_fill(one_hot.bool(), 0.0)
                non_gt_target = targets.masked_fill(one_hot.bool(), 0.0)
                target_entropy = -(targets.clamp_min(1e-12) * targets.clamp_min(1e-12).log()).sum(dim=-1)
                same_attr_non_gt = attr_match.bool() & ~one_hot.bool()
                same_obj_non_gt = obj_match.bool() & ~one_hot.bool()
                print(
                    '[pair_affinity_diag] '
                    f'eps={eps:.4f} tau={tau:.4f} exclude_gt={getattr(self.args, "affinity_exclude_gt", False)} '
                    f'batch={pair_id.shape[0]} pairs={num_pairs} '
                    f'gt_affinity_prob={gt_affinity_prob.mean().item():.6f} '
                    f'gt_target_prob={gt_target_prob.mean().item():.6f} '
                    f'non_gt_target_mass={non_gt_target.sum(dim=-1).mean().item():.6f} '
                    f'top_non_gt_target={non_gt_target.max(dim=-1).values.mean().item():.6f} '
                    f'same_attr_non_gt_mass={(non_gt_target * same_attr_non_gt).sum(dim=-1).mean().item():.6f} '
                    f'same_obj_non_gt_mass={(non_gt_target * same_obj_non_gt).sum(dim=-1).mean().item():.6f} '
                    f'target_entropy={target_entropy.mean().item():.6f} '
                    f'non_gt_affinity_mass={non_gt_affinity.sum(dim=-1).mean().item():.6f}',
                    flush=True,
                )
            self._affinity_diagnostics_logged = True

        return targets

    def forward(self, data):
        img, att_id, obj_id , pair_id = data[0] , data[1], data[2] , data[3] 

        # Obtain the prompt representations.
        token_com,token_att,token_obj = self.csp.construct_token_tensors(self.train_pairs)
        com_prompt_reps,att_prompt_reps,obj_prompt_reps = self.get_text_features(token_com,token_att,token_obj)

        # Align the four branches, optionally retaining class-specific latent
        # evidence instead of discarding it with unconditional mean pooling.
        use_route_aux = (
            getattr(self.args, 'sler_route_weight', 0.0) > 0 or
            getattr(self.args, 'sler_counterfactual_weight', 0.0) > 0
        )
        att_branch, obj_branch, com_branch, glb_branch, route_info = self.compute_branch_logits(
            img, com_prompt_reps, att_prompt_reps, obj_prompt_reps,
            return_routes=getattr(self.args, 'use_sler', False) and use_route_aux,
        )
        
        pair_targets = self.build_pair_affinity_targets(pair_id, att_id, obj_id, com_prompt_reps)
        att_loss = F.cross_entropy(att_branch , att_id)
        obj_loss = F.cross_entropy(obj_branch , obj_id)
        if getattr(self.args, 'use_pair_affinity', False):
            com_loss = self.soft_cross_entropy(com_branch, pair_targets)
            glb_loss = self.soft_cross_entropy(glb_branch, pair_targets)
        else:
            com_loss = F.cross_entropy(com_branch , pair_id)
            glb_loss = F.cross_entropy(glb_branch , pair_id)
                     
        route_loss = att_loss.new_zeros(())
        counterfactual_loss = att_loss.new_zeros(())
        if route_info is not None:
            route_loss, counterfactual_loss = self.sler_auxiliary_losses(
                att_id, obj_id, route_info
            )

        marginal_loss = att_loss.new_zeros(())
        consistency_loss = att_loss.new_zeros(())
        if getattr(self.args, 'use_smc', False):
            marginal_loss, consistency_loss = self.structured_marginal_losses(
                att_branch, obj_branch, com_branch, glb_branch, att_id, obj_id
            )

        total_loss = self.args.att_loss_weight * att_loss + \
                     self.args.obj_loss_weight * obj_loss + \
                     self.args.com_loss_weight * com_loss + \
                     self.args.glb_loss_weight * glb_loss + \
                     getattr(self.args, 'sler_route_weight', 0.0) * route_loss + \
                     getattr(self.args, 'sler_counterfactual_weight', 0.0) * counterfactual_loss + \
                     getattr(self.args, 'smc_marginal_weight', 0.0) * marginal_loss + \
                     getattr(self.args, 'smc_distill_weight', 0.0) * consistency_loss

        self.last_loss_dict = {
            'att': float(att_loss.detach().cpu()),
            'obj': float(obj_loss.detach().cpu()),
            'com': float(com_loss.detach().cpu()),
            'glb': float(glb_loss.detach().cpu()),
            'route': float(route_loss.detach().cpu()),
            'counterfactual': float(counterfactual_loss.detach().cpu()),
            'marginal': float(marginal_loss.detach().cpu()),
            'consistency': float(consistency_loss.detach().cpu()),
            'total': float(total_loss.detach().cpu()),
        }

        return total_loss
    
    def get_text_features(self,token_com,token_att,token_obj):
        text_features_com = self.text_encoder(
            self.csp.token_ids[0],
            token_com,
            enable_pos_emb=True
        )  
        text_features_att = self.text_encoder(
            self.csp.token_ids[1],
            token_att,
            enable_pos_emb=True
        )  
        text_features_obj = self.text_encoder(
            self.csp.token_ids[2],
            token_obj,
            enable_pos_emb=True
        )  
        text_features_com = F.normalize(text_features_com, p=2, dim=-1)  
        text_features_att = F.normalize(text_features_att, p=2, dim=-1)  
        text_features_obj = F.normalize(text_features_obj, p=2, dim=-1)  
        return text_features_com,text_features_att,text_features_obj
    
