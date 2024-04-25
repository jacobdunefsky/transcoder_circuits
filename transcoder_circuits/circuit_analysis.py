# --- code for circuit analysis --- #

import torch
import numpy as np
import tqdm

from transformer_lens.utils import get_act_name, to_numpy

# first, some preliminary functions, used to calculate
#   attribs between pairs of components

@torch.no_grad()
def get_attn_head_contribs(model, cache, layer, range_normal):
	split_vals = cache[get_act_name('v', layer)]
	attn_pattern = cache[get_act_name('pattern', layer)]
	#'batch head dst src, batch src head d_head -> batch head dst src d_head'
	weighted_vals = torch.einsum(
		'b h d s, b s h f -> b h d s f',
		attn_pattern, split_vals
	)

  # 'batch head dst src d_head, head d_head d_model -> batch head dst src d_model'
	weighted_outs = torch.einsum(
		'b h d s f, h f m -> b h d s m',
		weighted_vals, model.W_O[layer]
	)

  # 'batch head dst src d_model, d_model -> batch head dst src'
	contribs = torch.einsum(
		'b h d s m, m -> b h d s',
		weighted_outs, range_normal
	)

	return contribs

@torch.no_grad()
def get_transcoder_ixg(transcoder, cache, range_normal, input_layer, input_token_idx, return_numpy=True, is_transcoder_post_ln=True, return_feature_activs=True):
    pulledback_feature = transcoder.W_dec @ range_normal
    if is_transcoder_post_ln:
        act_name = get_act_name('normalized', input_layer, 'ln2')
    else:
        act_name = get_act_name('resid_mid', input_layer)

    feature_activs = transcoder(cache[act_name])[1][0,input_token_idx]
    pulledback_feature = pulledback_feature * feature_activs
    if return_numpy:
        pulledback_feature = to_numpy(pulledback_feature)
        feature_activs = to_numpy(feature_activs)

    if not return_feature_activs:
        return pulledback_feature
    else:
        return pulledback_feature, feature_activs

# get the mean input-times-gradient vector over a dataset of tokens
@torch.no_grad()
def get_mean_ixg(model, tokens_arr, range_transcoder, range_feature_idx, transcoder, token_idxs=None, batch_size=64, do_sum_count=False):
    act_name = transcoder.cfg.hook_point
    layer = transcoder.cfg.hook_point_layer

    range_normal = range_transcoder.W_enc[:, range_feature_idx]
    pulledback_feature = transcoder.W_dec @ range_normal

    
    if token_idxs is None:
        tokens_gen = tqdm.tqdm(range(0, tokens_arr.shape[0], batch_size))
    else:
        tokens_gen = tqdm.tqdm(token_idxs)
    
    if not do_sum_count:
        mean_ixgs = []
    else:
        ixgs_sum = np.zeros(transcoder.W_enc.shape[1])
        ixgs_count = np.zeros(transcoder.W_enc.shape[1])

    for t in tokens_gen:
        if token_idxs is not None:
            example_idx, token_idx = t
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens_arr[example_idx, :token_idx+1], stop_at_layer=layer+1, names_filter=[
                    act_name
                ])
                acts = cache[act_name]
                feature_activs = transcoder(acts)[1][0, token_idx]
                cur_ixg = (pulledback_feature * feature_activs)[None]
        else:
            i = t
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens_arr[i:i+batch_size], stop_at_layer=layer+1, names_filter=[
                    act_name
                ])
                acts = cache[act_name]
                feature_activs = transcoder(acts)[1].reshape(-1, transcoder.W_enc.shape[1])
                
                cur_ixg = torch.einsum('i, ji -> ji', pulledback_feature, feature_activs)

        if not do_sum_count:
            mean_ixgs.append(np.mean(to_numpy(cur_ixg), axis=0))
        else:
            ixgs_sum += to_numpy(cur_ixg).sum(axis=0)
            ixgs_count += np.abs(to_numpy(cur_ixg)>0).sum(axis=0)

    if do_sum_count:
        ixgs_count[ixgs_count == 0] = 1
        return ixgs_sum/ixgs_count, ixgs_count/len(token_idxs)
    else:
        return np.mean(mean_ixgs, axis=0)

# approximate layernorms as constants when propagating feature vectors backward
# for theoretical motivation, see the LayerNorm section of
#	https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
@torch.no_grad()
def get_ln_constant(model, cache, vector, layer, token, is_ln2=False, recip=False):
    x_act_name = get_act_name('resid_mid', layer) if is_ln2 else get_act_name('resid_pre', layer)
    x = cache[x_act_name][0, token]

    y_act_name = get_act_name('normalized', layer, 'ln2' if is_ln2 else 'ln1')
    y = cache[y_act_name][0, token]

    if torch.dot(vector, x) == 0:
        return torch.tensor(0.)
    return torch.dot(vector, y)/torch.dot(vector, x) if not recip else torch.dot(vector, x)/torch.dot(vector, y)

# now, we'll actually write the circuit analysis code that finds
#  computational paths and calculates their importances

import enum
from dataclasses import dataclass, field
from typing import Optional, List
import copy

# define some classes

class ComponentType(enum.Enum):
    MLP = 'mlp'
    ATTN = 'attn'
    EMBED = 'embed'
    
    # error terms
    TC_ERROR = 'tc_error' # error due to inaccurate transcoders
    PRUNE_ERROR = 'prune_error' # error due to only looking at top paths in graph
    BIAS_ERROR = 'bias_error' # account for bias terms in transcoders

class FeatureType(enum.Enum):
    NONE = 'none'
    SAE = 'sae'
    TRANSCODER = 'tc'

class ContribType(enum.Enum):
    RAW = 'raw'
    ZERO_ABLATION = 'zero_ablation'

# Component: an individual component (e.g. an attn head or a transcoder feature)
@dataclass
class Component:
    layer: int
    component_type: ComponentType

    token: Optional[int] = None

    attn_head: Optional[int] = None

    feature_type: Optional[FeatureType] = None
    feature_idx: Optional[int] = None

    def __str__(self, show_token=True):
        retstr = ''
        feature_type_str = ''

        base_str = f'{self.component_type.value}{self.layer}'
        attn_str = '' if self.component_type != ComponentType.ATTN else f'[{self.attn_head}]'
        
        feature_str = ''
        if self.feature_type is not None and self.feature_idx is not None:
            feature_str = f"{self.feature_type.value}[{self.feature_idx}]"
            
        token_str = ''
        if self.token is not None and show_token:
            token_str = f'@{self.token}'

        retstr = ''.join([base_str, attn_str, feature_str, token_str])
        return retstr

    def __repr__(self):
        return f'<Component object {str(self)}>'

# FeatureVector: a unique feature vector potentially associated with a path of components
#  along with a contrib
@dataclass
class FeatureVector:
    # a list of components that can be used to uniquely specify the direction of the feature vector
    component_path: List[Component]
    # TODO: potentially add "gradients" and "activations" lists

    vector: torch.Tensor

    # sublayer can be 'mlp_out', 'resid_post', 'resid_mid', 'resid_pre'
    # denotes where the feature vector lives
    layer: int
    sublayer: str
    token: Optional[int] = None

    contrib: Optional[float] = None
    contrib_type: Optional[ContribType] = None

    error: float = 0.0

    def __post_init__(self):
        if self.token is None and len(self.component_path)>0: self.token = self.component_path[-1].token
        if self.layer is None and len(self.component_path)>0: self.layer = self.component_path[-1].layer

    # note: str(FeatureVector) should return a string that uniquely identifies a feature direction (e.g. for use in a causal graph)
    # (this is distinct from a unique feature *vector*, by the way)
    def __str__(self, show_full=True, show_contrib=True, show_last_token=True):
        retstr = ''
        token_str = '' if self.token is None or not show_last_token else f'@{self.token}'
        if len(self.component_path) > 0:
            if show_full:
                retstr = ''.join(x.__str__(show_token=False) for x in self.component_path[:-1])
            retstr = ''.join([retstr, self.component_path[-1].__str__(show_token=False), token_str])
        else:
            retstr = f'*{self.sublayer}{self.layer}{token_str}'
        if show_contrib and self.contrib is not None:
            retstr = ''.join([retstr, f': {self.contrib:.2}'])
        return retstr

    def __repr__(self):
        contrib_type_str = '' if self.contrib_type is None else f' contrib_type={self.contrib_type.value}'
        return f'<FeatureVector object {str(self)}, sublayer={self.sublayer}{contrib_type_str}>'

@torch.no_grad()
def make_sae_feature_vector(sae, feature_idx, use_encoder=True, token=-1):
    hook_point = sae.cfg.hook_point if (use_encoder or not sae.cfg.is_transcoder) else sae.cfg.out_hook_point
    layer = sae.cfg.hook_point_layer if (use_encoder or not sae.cfg.is_transcoder) else sae.cfg.out_hook_point_layer
    feature_type = FeatureType.SAE if not sae.cfg.is_transcoder else FeatureType.TRANSCODER
    vector = sae.W_enc[:,feature_idx] if use_encoder else sae.W_dec[feature_idx]
    vector = torch.clone(vector.detach())
    vector.requires_grad = False
    vector.requires_grad_(False)
    if 'resid_mid' in hook_point or ('normalized' in hook_point and 'ln2' in hook_point):
        # currently, we treat ln2normalized as resid_mid
        # this is kinda ugly, but because we account for layernorm constants in later
        #  functions, this does work now
        sublayer = 'resid_mid'
        component_type = ComponentType.MLP
    elif 'resid_pre' in hook_point:
        sublayer = 'resid_pre'
        component_type = ComponentType.ATTN
    elif 'mlp_out' in hook_point:
        sublayer = 'mlp_out'
        component_type = ComponentType.MLP
    elif 'resid_post' in hook_point:
        sublayer = 'resid_post'
        component_type = ComponentType.ATTN

    my_feature = FeatureVector(
        component_path=[Component(
            layer=layer,
            component_type=component_type,
            token=token,
            feature_type=feature_type,
            feature_idx=feature_idx
        )],
        
        layer = layer,
        sublayer = sublayer,
        vector = vector
    )

    return my_feature

@torch.no_grad()
def get_top_transcoder_features(model, transcoder, cache, feature_vector, layer, k=5):
    my_token = feature_vector.token if feature_vector.token >= 0 else cache[get_act_name('resid_pre', 0)].shape[1]+feature_vector.token
    is_transcoder_post_ln = 'ln2' in transcoder.cfg.hook_point and 'normalized' in transcoder.cfg.hook_point
    
    # compute error
    if is_transcoder_post_ln:
        act_name = get_act_name('normalized', layer, 'ln2')
    else:
        act_name = get_act_name('resid_mid', layer)
    transcoder_out = transcoder(cache[act_name])[0][0,my_token]
    mlp_out = model.blocks[layer].mlp(cache[act_name])[0, my_token]
    error = torch.dot(feature_vector.vector, mlp_out - transcoder_out)/torch.dot(feature_vector.vector, mlp_out)

    # compute pulledback feature
    pulledback_feature, feature_activs = get_transcoder_ixg(transcoder, cache, feature_vector.vector, layer, feature_vector.token, return_numpy=False, is_transcoder_post_ln=is_transcoder_post_ln)
    top_contribs, top_indices = torch.topk(pulledback_feature, k=k)

    top_contribs_list = []
    for contrib, index in zip(top_contribs, top_indices):
        vector = transcoder.W_enc[:, index]
        vector = vector * (transcoder.W_dec @ feature_vector.vector)[index]

        if is_transcoder_post_ln:
            vector = vector * get_ln_constant(model, cache, vector, layer, feature_vector.token, is_ln2=True)

        new_component = Component(
            layer=layer,
            component_type=ComponentType.MLP,
            token=my_token,
            feature_type=FeatureType.TRANSCODER,
            feature_idx=index.item(),
        )
        top_contribs_list.append(FeatureVector(
            component_path=[new_component],
            vector = vector,
            layer=layer,
            sublayer="resid_mid",
            contrib=contrib.item(),
            contrib_type=ContribType.RAW,
            error=error,
        ))
    return top_contribs_list

@torch.no_grad()
def get_top_contribs(model, transcoders, cache, feature_vector, k=5, ignore_bos=True, only_return_all_scores=False, cap=None, filter=None):
    if feature_vector.sublayer == "mlp_out":
        return get_top_transcoder_features(model, transcoders[feature_vector.layer], cache, feature_vector, feature_vector.layer, k=k)

    my_layer = feature_vector.layer
    
    # get MLP contribs
    all_mlp_contribs = []
    mlp_max_layer = my_layer + (1 if feature_vector.sublayer == 'resid_post' else 0)
    for cur_layer in range(mlp_max_layer):
        cur_top_features = get_top_transcoder_features(model, transcoders[cur_layer], cache, feature_vector, cur_layer, k=k)
        all_mlp_contribs = all_mlp_contribs + cur_top_features

    # get attn contribs
    all_attn_contribs = []
    attn_max_layer = my_layer + (1 if feature_vector.sublayer == 'resid_post' or feature_vector.sublayer == 'resid_mid' else 0)
    for cur_layer in range(attn_max_layer):
        attn_contribs = get_attn_head_contribs(model, cache, cur_layer, feature_vector.vector)[0, :, feature_vector.token, :]
        if ignore_bos:
            attn_contribs = attn_contribs[:, 1:]
        top_attn_contribs_flattened, top_attn_contrib_indices_flattened = torch.topk(attn_contribs.flatten(), k=np.min([k, len(attn_contribs)]))
        top_attn_contrib_indices = np.array(np.unravel_index(to_numpy(top_attn_contrib_indices_flattened), attn_contribs.shape)).T

        for contrib, (head, src_token) in zip(top_attn_contribs_flattened, top_attn_contrib_indices):
            if ignore_bos:
                src_token = src_token + 1
            vector = model.OV[cur_layer, head] @ feature_vector.vector
            attn_pattern = cache[get_act_name('pattern', cur_layer)]
            vector = vector * attn_pattern[0, head, feature_vector.token, src_token]
            ln_constant = get_ln_constant(model, cache, vector, cur_layer, src_token, is_ln2=False)
            vector = vector * ln_constant
            if ln_constant.isnan(): print("Nan!")

            new_component = Component(
                layer=cur_layer,
                component_type=ComponentType.ATTN,
                token=src_token,
                attn_head=head
            )
            new_feature_vector = FeatureVector(
                component_path=feature_vector.component_path + [new_component],
                vector=vector,
                layer=cur_layer,
                sublayer='resid_pre',
                contrib=contrib.item(),
                contrib_type=ContribType.RAW
            )
            all_attn_contribs.append(new_feature_vector)

    # get embedding contribs
    my_token = feature_vector.token if feature_vector.token >= 0 else cache[get_act_name('resid_pre', 0)].shape[1]+feature_vector.token
    embedding_contrib = FeatureVector(
        component_path = feature_vector.component_path + [Component(
            layer=0,
            component_type=ComponentType.EMBED,
            token=my_token,
        )],
        vector=feature_vector.vector,
        layer=0,
        sublayer='resid_pre',
        contrib=torch.dot(cache[get_act_name('resid_pre', 0)][0, feature_vector.token], feature_vector.vector).item(),
        contrib_type=ContribType.RAW
    )

    # get top contribs from all categories
    all_contribs = all_mlp_contribs + all_attn_contribs + [embedding_contrib]

    if filter is not None:
        all_contribs = [ x for x in all_contribs if filter.match(x) ]
    
    if cap is not None:
        for i, contrib in enumerate(all_contribs):
            if contrib.contrib > cap:
                all_contribs[i].contrib = cap
                all_contribs[i].contrib_type = ContribType.ZERO_ABLATION
    all_contrib_scores = torch.tensor([x.contrib for x in all_contribs])
    if only_return_all_scores: return all_contrib_scores
    
    _, top_contrib_indices = torch.topk(all_contrib_scores, k=np.min([k, len(all_contrib_scores)]))
    return [all_contribs[i.item()] for i in top_contrib_indices]

@torch.no_grad()
def greedy_get_top_paths(model, transcoders, cache, feature_vector, num_iters=2, num_branches=5, ignore_bos=True, do_raw_attribution=False, filter=None):
    do_cap = not do_raw_attribution # historical name change; TODO: refactor
    
    all_paths = []
    new_root = copy.deepcopy(feature_vector)

    # deal with LN constant
    # TODO: this is hacky and makes the assumption that if feature_vector is a transcoder feature, then it comes from the passed list of transcoders
    if new_root.component_path[-1].feature_type == FeatureType.TRANSCODER:
        tc = transcoders[new_root.layer]
        if 'ln2.hook_normalized' in tc.cfg.hook_point:
            ln_constant = get_ln_constant(model, cache, new_root.vector, new_root.layer, new_root.token, is_ln2=True)
            new_root.vector *= ln_constant
        new_root.contrib = tc(cache[tc.cfg.hook_point])[1][0, new_root.token, new_root.component_path[-1].feature_idx].item()
    cur_paths = [[new_root]]
    for iter in range(num_iters):
        new_paths = []
        for path in cur_paths:
            cur_feature = path[-1]
            if cur_feature.layer == 0 and cur_feature.sublayer == 'resid_pre': continue
            
            cap = None
            if do_cap:
                # Cap feature contribs at smallest transcoder feature activation
                # This corresponds to calculating feature attribs by
                #   zero-ablating the output of the feature
                for cap_feature in path:
                    if len(cap_feature.component_path) > 0 and (cap_feature.component_path[-1].feature_type == FeatureType.TRANSCODER or cap_feature.component_path[-1].feature_type == FeatureType.SAE and (cap is None or cap_feature.contrib < cap)):
                        cap = cap_feature.contrib
            
            cur_top_contribs = get_top_contribs(model, transcoders, cache, cur_feature, k=num_branches, ignore_bos=ignore_bos, cap=cap, filter=filter)
            new_paths = new_paths + [ path + [cur_top_contrib] for cur_top_contrib in cur_top_contribs ]
        _, top_new_path_indices = torch.topk(torch.tensor([new_path[-1].contrib for new_path in new_paths]), k=np.min([num_branches, len(new_paths)]))
        cur_paths = [ new_paths[i] for i in top_new_path_indices ]
        all_paths.append(cur_paths)
    return all_paths

def print_all_paths(paths):
    if len(paths) == 0: return
    if type(paths[0][0]) is list:
        for i, cur_paths in enumerate(paths):
            try:
                print(f"--- Paths of size {len(cur_paths[0])} ---")
            except:
                continue
            for j, cur_path in enumerate(cur_paths):
                print(f"Path [{i}][{j}]: ", end="")
                print(" <- ".join(map(lambda x: x.__str__(show_full=False, show_last_token=True), cur_path)))
    else:
        for j, cur_path in enumerate(paths):
            print(f"Path [{j}]: ", end="")
            print(" <- ".join(map(lambda x: x.__str__(show_full=False, show_last_token=True), cur_path)))

@torch.no_grad()
def get_raw_top_features_among_paths(all_paths, use_tokens=True, top_k=5, filter_layers=None, filter_sublayers=None):
    retdict = {}
    str_to_feature = {}
    for i, cur_length_paths in enumerate(all_paths):
        for j, cur_path in enumerate(cur_length_paths):
            cur_feature = cur_path[-1]
            if filter_layers is not None and cur_feature.layer not in filter_layers: continue
            if filter_sublayers is not None and cur_feature.sublayer not in filter_sublayers: continue

            cur_feature_str = cur_feature.__str__(show_contrib=False, show_last_token=use_tokens)

            contrib = 0
            if cur_feature.contrib is not None:
                assert(cur_feature.contrib_type == ContribType.RAW)
                contrib = cur_feature.contrib
            if cur_feature_str not in retdict:
                try:
                    retdict[cur_feature_str] = copy.deepcopy(cur_feature)
                except Exception as e:
                    print(cur_feature)
                    print(cur_feature.component_path)
                    print(cur_feature.vector)
                    raise e
                retdict[cur_feature_str].contrib = contrib
            else:
                retdict[cur_feature_str].contrib = retdict[cur_feature_str].contrib + contrib
    if top_k is None or top_k > len(retdict): top_k = len(retdict)
    top_scores, top_indices = torch.topk(torch.tensor([x.contrib for x in retdict.values()], dtype=torch.float), k=top_k)
    retlist = []
    keys_list = list(retdict.keys())
    for score, index in zip(top_scores, top_indices):
        cur_feature = retdict[keys_list[index.item()]]
        if not use_tokens:
            for component in cur_feature.component_path: component.token=None
            cur_feature.token=None
        retlist.append(cur_feature)
    return retlist

# now, code for filtering computational paths

import dataclasses
class FilterType(enum.Enum):
    EQ = enum.auto() # equals
    NE = enum.auto() # not equal to
    GT = enum.auto() # greater than
    GE = enum.auto() # greater than or equal to
    LT = enum.auto() # less than 
    LE = enum.auto() # less than or equal to

@dataclass
class FeatureFilter:
    # feature-level filters
    layer: Optional[int] = field(default=None, metadata={'filter_level': 'feature'})
    layer_filter_type: FilterType = FilterType.EQ
    sublayer: Optional[int] = field(default=None, metadata={'filter_level': 'feature'})
    sublayer_filter_type: FilterType = FilterType.EQ
    token: Optional[int] = field(default=None, metadata={'filter_level': 'feature'})
    token_filter_type: FilterType = FilterType.EQ

    # filters on last component in component_path
    component_type: Optional[ComponentType] = field(default=None, metadata={'filter_level': 'component'})
    component_type_filter_type: FilterType = FilterType.EQ
    attn_head: Optional[int] = field(default=None, metadata={'filter_level': 'component'})
    attn_head_filter_type: FilterType = FilterType.EQ
    feature_type: Optional[FeatureType] = field(default=None, metadata={'filter_level': 'component'})
    feature_type_filter_type: FilterType = FilterType.EQ
    feature_idx: Optional[int] = field(default=None, metadata={'filter_level': 'component'}) 
    feature_idx_filter_type: FilterType = FilterType.EQ       

    def match(self, feature):
        component = None
        
        for field in dataclasses.fields(self):
            name = field.name
            val = self.__dict__[name]
            if val is None: continue
            
            try:
                filter_level = field.metadata['filter_level']
            except KeyError:
                continue # not a filter
            if filter_level == 'feature':
                if val is not None:
                    filter_type = self.__dict__[f'{name}_filter_type']
                    if filter_type == FilterType.EQ and val != feature.__dict__[name]: return False
                    if filter_type == FilterType.NE and val == feature.__dict__[name]: return False
                    if filter_type == FilterType.GT and feature.__dict__[name] <= val: return False
                    if filter_type == FilterType.GE and feature.__dict__[name] < val: return False
                    if filter_type == FilterType.LT and feature.__dict__[name] >= val: return False
                    if filter_type == FilterType.LE and feature.__dict__[name] > val: return False
            elif filter_level == 'component':
                if component is None:
                    if len(feature.component_path) <= 0: return False
                    component = feature.component_path[-1]
                if val is not None:
                    filter_type = self.__dict__[f'{name}_filter_type']
                    if filter_type == FilterType.EQ and val != component.__dict__[name]: return False
                    if filter_type == FilterType.NE and val == component.__dict__[name]: return False
        return True

import functools
def flatten_nested_list(x):
    return list(functools.reduce(lambda a,b: a+b, x))

def get_paths_via_filter(all_paths, infix_path=None, not_infix_path=None, suffix_path=None):
    retpaths = []
    if type(all_paths[0][0]) is list:
        path_list = flatten_nested_list(all_paths)
    else:
        path_list = all_paths
    for path in path_list:
        if not_infix_path is not None:
            if len(path) < len(not_infix_path): continue

            match_started = False
            path_good = True
            i = 0
            for j, cur_feature in enumerate(path):
                cur_infix_filter = not_infix_path[i]
                
                if cur_infix_filter.match(cur_feature):
                    if not match_started:
                        if len(path[j:]) < len(not_infix_path): break
                        match_started = True
                elif match_started:
                    path_good = False
                    break
                    
                if match_started:
                    i = i + 1
                    if i >= len(not_infix_path): break
            if not (match_started and path_good): retpaths.append(path)
        
        if infix_path is not None:
            if len(path) < len(infix_path): continue

            match_started = False
            path_good = True
            i = 0
            for j, cur_feature in enumerate(path):
                cur_infix_filter = infix_path[i]
                
                if cur_infix_filter.match(cur_feature):
                    if not match_started:
                        if len(path[j:]) < len(infix_path): break
                        match_started = True
                elif match_started:
                    path_good = False
                    break
                    
                if match_started:
                    i = i + 1
                    if i >= len(infix_path): break
            if match_started and path_good: retpaths.append(path)
        
        if suffix_path is not None:
            if len(path) < len(suffix_path): continue
            path_good = True
            for i in range(1, len(suffix_path)+1):
                cur_feature = path[-i]
                cur_suffix_filter = suffix_path[-i]
                if not cur_suffix_filter.match(cur_feature):
                    path_good = False
                    break
            if path_good: retpaths.append(path)
    return retpaths

# code for combining computational paths into graphs

def path_to_str(path, show_contrib=False, show_last_token=False):
    return " <- ".join(list(x.__str__(show_contrib=show_contrib, show_last_token=show_last_token) for x in path))

import collections

@torch.no_grad()
def paths_to_graph(all_paths):
    if type(all_paths[0][0]) is list:
        path_list = flatten_nested_list(all_paths)
    else:
        path_list = all_paths
        
    retdict = collections.defaultdict(int)
    nodes = {}
    seen_prefixes = set()
    for i, cur_path in enumerate(path_list):
        for j in range(0, len(cur_path)):
            prefix = cur_path[:j+1]
            prefix_str = path_to_str(prefix, show_last_token=True)
            if prefix_str in seen_prefixes: continue
            seen_prefixes.add(prefix_str)

            if j == 0:
                # prefix is of size 1
                try:
                    my_contrib = prefix[0].contrib if prefix[0].contrib is not None else 0
                    nodes[prefix_str].contrib += my_contrib
                    if prefix[0].contrib is not None and my_contrib != 0:
                        nodes[prefix_str].vector = nodes[prefix_str].vector + prefix[0].vector
                except KeyError:
                    nodes[prefix_str] = copy.copy(prefix[0])
                continue
            
            parent, child = cur_path[j-1], cur_path[j]
            parent_str = parent.__str__(show_contrib=False, show_last_token=True, show_full=False)
            child_str = child.__str__(show_contrib=False, show_last_token=True, show_full=False)
            assert child.contrib_type == ContribType.RAW, f"Contrib type is not ContribType.RAW: {child_str}->{parent_str}"
            retdict[(child_str, parent_str)] += child.contrib

            try:
                nodes[child_str].contrib += child.contrib
                nodes[child_str].vector = nodes[child_str].vector + child.vector
            except KeyError:
                nodes[child_str] = copy.deepcopy(child)

    # last step: go through all the attention nodes and trim their component_paths
    # (this is because they now correspond to multiple later-layer features)

    new_nodes = { node_str: nodes[node_str] for node_str in nodes }
    for node_str in nodes:
        node = nodes[node_str]
        if node.component_path[-1].component_type != ComponentType.ATTN: continue

        node.component_path = [node.component_path[-1]]
        try:
            del new_nodes[node_str]
        except:
            pass
        new_nodes[node.__str__(show_contrib=False, show_last_token=True, show_full=False)] = node
    
    return retdict, new_nodes

@torch.no_grad()
def add_error_nodes_to_graph(model, cache, transcoders, edges, nodes, do_bias=True):
    # add error nodes representing error in transcoders and error due to computational paths being pruned

    # first: deal with transcoder error
    new_edges = { edge: edges[edge] for edge in edges }
    new_nodes = { node_str: nodes[node_str] for node_str in nodes }
    # only want to add error nodes to non-leaf nodes
    # also, fill up a dict of nodes' children for later
    children_dict = {}
    for child_str, parent_str in edges:
        if parent_str in children_dict:
            children_dict[parent_str].append(child_str)
            continue
        else:
            children_dict[parent_str] = [child_str]
        
        parent = nodes[parent_str]
        if parent.sublayer not in ['resid_mid', 'resid_pre', 'resid_post']: continue

        max_layer = parent.layer
        if parent.sublayer == 'resid_post': max_layer = max_layer + 1

        error = 0.
        #print(max_layer, parent.sublayer)
        for layer in range(max_layer):
            mlp_out = cache[get_act_name('mlp_out', layer)][0, parent.token]
            tc = transcoders[layer]
            tc_out = tc(cache[tc.cfg.hook_point])[0][0, parent.token]
            error += torch.dot(parent.vector, mlp_out - tc_out).item()
        error_feature = FeatureVector(
            component_path=[parent.component_path[-1], Component(
                layer=parent.layer,
                component_type=ComponentType.TC_ERROR,
                token=parent.token,
            )],
            
            layer = parent.layer,
            sublayer = parent.sublayer,
            vector = None, # TODO: deal with conflicting type annotation

            contrib = error,
        )
        error_str = error_feature.__str__(show_contrib=False, show_last_token=True, show_full=True)
        new_edges[(error_str, parent_str)] = error
        new_nodes[error_str] = error_feature

    edges = new_edges
    nodes = new_nodes

    # next, deal with bias term error
    if do_bias:
        new_edges = { edge: edges[edge] for edge in edges }
        new_nodes = { node_str: nodes[node_str] for node_str in nodes }
        for parent_str in children_dict:
            #print(parent_str)
            parent = nodes[parent_str]
            if parent.component_path[-1].feature_type not in [FeatureType.TRANSCODER]: continue #[FeatureType.TRANSCODER, FeatureType.SAE]
            bias = (-transcoders[parent.layer].W_enc[:, parent.component_path[-1].feature_idx]\
                @ transcoders[parent.layer].b_dec\
                + transcoders[parent.layer].b_enc[ parent.component_path[-1].feature_idx]).item()
            
            bias_feature = FeatureVector(
                component_path=[parent.component_path[-1], Component(
                    layer=parent.layer,
                    component_type=ComponentType.BIAS_ERROR,
                    token=parent.token,
                )],
                
                layer = parent.layer,
                sublayer = parent.sublayer,
                vector = None, # TODO: deal with conflicting type annotation
    
                contrib = bias,
            )
            bias_str = bias_feature.__str__(show_contrib=False, show_last_token=True, show_full=True)
            new_edges[(bias_str, parent_str)] = bias
            new_nodes[bias_str] = bias_feature
        
        edges = new_edges
        nodes = new_nodes

    # next, deal with pruning error
    new_edges = { edge: edges[edge] for edge in edges }
    new_nodes = { node_str: nodes[node_str] for node_str in nodes }
    for parent_str in children_dict:
        edge_contribs = 0.
        for child_str in children_dict[parent_str]:
            edge_contribs += edges[(child_str, parent_str)]
        parent = nodes[parent_str]
        error = parent.contrib - edge_contribs
        error_feature = FeatureVector(
            component_path=[parent.component_path[-1], Component(
                layer=parent.layer,
                component_type=ComponentType.PRUNE_ERROR,
                token=parent.token,
            )],
            
            layer = parent.layer,
            sublayer = parent.sublayer,
            vector = None, # TODO: deal with conflicting type annotation

            contrib = error,
        )
        error_str = error_feature.__str__(show_contrib=False, show_last_token=True, show_full=True)
        new_edges[(error_str, parent_str)] = error
        new_nodes[error_str] = error_feature

    return new_edges, new_nodes    

def sum_over_tokens(edges, nodes):
    new_edges = collections.defaultdict(int)
    for (child_str, parent_str), score in edges.items():
        new_child_str = nodes[child_str].__str__(show_contrib=False, show_last_token=False, show_full=False)
        new_parent_str = nodes[parent_str].__str__(show_contrib=False, show_last_token=False, show_full=False)
        new_edges[new_child_str, new_parent_str] += score
    
    new_nodes = {}
    for node in nodes.values():
        node_str = node.__str__(show_contrib=False, show_last_token=False, show_full=False)
        try:
            new_nodes[node_str].contrib += node.contrib
        except KeyError:
            new_nodes[node_str] = copy.copy(node)

    return new_edges, new_nodes

# graph plotting code
# NOTE: this code doesn't produce the nicest looking graphs

import plotly.graph_objs as go
import collections

def layer_to_float(feature):
    layer = feature.layer
    if feature.sublayer == 'resid_mid': layer = layer + 0.5
    if feature.component_path[-1].component_type in [ComponentType.PRUNE_ERROR, ComponentType.BIAS_ERROR, ComponentType.TC_ERROR]:
        layer = layer - 0.25
    return layer

def nodes_to_coords(nodes, y_jitter=0.3, y_mult=1.0):
    retdict = {}
    num_nodes_in_xval = collections.defaultdict(int)
    for node_name, feature in nodes.items():
        xval = layer_to_float(feature)
        retdict[node_name] = [xval, num_nodes_in_xval[xval]]
        num_nodes_in_xval[xval] += 1
    for node_name in retdict:
        num_nodes = num_nodes_in_xval[retdict[node_name][0]]
        retdict[node_name][1] = retdict[node_name][1]/(num_nodes-1) if num_nodes != 1 else 0.5
        if num_nodes != 1: 
            cur_y_noise = np.random.uniform(0, y_jitter)
            if retdict[node_name][1] > 0.5: cur_y_noise *= -1
            cur_y_noise /= (num_nodes-1)
        else:
            cur_y_noise = np.random.uniform(-y_jitter, y_jitter)
        retdict[node_name][1] += cur_y_noise
        retdict[node_name][1] *= y_mult
    return retdict

def get_contribs_in_graph(edges, nodes):
    new_nodes = {}
    for (child_str, parent_str), contrib in edges.items():
        try:
            new_nodes[parent_str].contrib += contrib
        except KeyError:
            new_nodes[parent_str] = copy.copy(nodes[parent_str])
            new_nodes[parent_str].contrib = contrib

    for node in nodes:
        if node not in new_nodes:
            new_nodes[node] = copy.copy(nodes[node])

    return new_nodes

def plot_graph(edges, nodes, y_mult=1.0, width=800, height=600, arrow_width_multiplier=3.0, only_get_contribs_in_graph=False):
    # TODO: ugly code reuse with feature_dashboard
    def batch_color_interpolate(scores, max_color, zero_color, scores_min=None, scores_max=None):
        if scores_min is None: scores_min = scores.min()
        if scores_max is None: scores_max = scores.max()
        scores_normalized = (scores - scores_min) / (scores_max - scores_min)
        
        max_color_vec = np.array([int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)])
        zero_color_vec = np.array([int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)])
    
        color_vecs = np.einsum('i, j -> ij', scores_normalized, max_color_vec) + np.einsum('i, j -> ij', 1-scores_normalized, zero_color_vec)
        color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
        return color_strs

    arrow_widths = np.array(list(edges.values()))
    arrow_widths = 1 + arrow_width_multiplier*(arrow_widths-arrow_widths.min())/(arrow_widths.max()-arrow_widths.min())

    layout = nodes_to_coords(nodes, y_mult=y_mult)
    colors = batch_color_interpolate(np.array([x.contrib if x.contrib is not None else 0 for x in nodes.values()]), '#22ff22', '#ffffff')
    
    trace = go.Scatter( 
        x=[val[0] for val in layout.values()], 
        y=[val[1] for val in layout.values()],
        hoverinfo='text',
        hovertext=[f"{key}<br>Contrib: {val.contrib:.2}" if val.contrib is not None else f"{key}" for key, val in nodes.items()],
        mode='markers',
        marker=dict(size=20, line=dict(width=1, color='Black'), color=colors)
    )

    trace2 = go.Scatter( 
        x=[(layout[edge[0]][0]+layout[edge[1]][0])/2 for edge in edges], 
        y=[(layout[edge[0]][1]+layout[edge[1]][1])/2 for edge in edges],
        hoverinfo='text',
        hovertext=[f"{edge[0]} -> {edge[1]}<br>Contrib: {contrib:.2}" if contrib is not None else f"{edge[0]} -> {edge[1]}" for edge, contrib in edges.items()],
        mode='markers', marker_symbol='square',
        marker=dict(size=5, line=dict(width=1, color='Black'), color='black')
    )
    
    # Plot edges
    # Thank you to https://stackoverflow.com/a/51430912 for the general idea
    x0, y0, x1, y1 = [], [], [], []
    
    for edge in edges:
        x0.append(layout[edge[0]][0])
        y0.append(layout[edge[0]][1])
        x1.append(layout[edge[1]][0])
        y1.append(layout[edge[1]][1])
    
    fig = go.Figure(
        data=[trace, trace2],
        layout=go.Layout(
            autosize=False,
            width=width,
            height=height,
            showlegend=False,
            xaxis_title='Layer',
            annotations = [
                dict(ax=x0[i], ay=y0[i], axref='x', ayref='y',
                    x=x1[i], y=y1[i], xref='x', yref='y',
                    showarrow=True, arrowhead=1, arrowwidth=arrow_widths[i]) for i in range(0, len(x0))
            ],
            yaxis = dict(
                tickmode = 'array',
                tickvals = [],
                ticktext = []
            )
        )
    ) 
    fig.show()