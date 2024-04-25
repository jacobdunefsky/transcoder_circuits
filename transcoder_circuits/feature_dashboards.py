# --- feature scores / feature dashboard functions --- #

import numpy as np
import tqdm
import torch

import matplotlib.pyplot as plt

from IPython.display import HTML
from transformer_lens.utils import get_act_name, to_numpy

def get_feature_scores(model, encoder, tokens_arr, feature_idx, batch_size=64, act_name='resid_pre', layer=0, use_raw_scores=False, use_decoder=False, feature_post=None, ignore_endoftext=False):
	act_name = encoder.cfg.hook_point
	layer = encoder.cfg.hook_point_layer
		
	scores = []
	endoftext_token = model.tokenizer.eos_token 
	for i in tqdm.tqdm(range(0, tokens_arr.shape[0], batch_size)):
		with torch.no_grad():
			_, cache = model.run_with_cache(tokens_arr[i:i+batch_size], stop_at_layer=layer+1, names_filter=[
				act_name
			])
			mlp_acts = cache[act_name]
			mlp_acts_flattened = mlp_acts.reshape(-1, encoder.W_enc.shape[0])
			if feature_post is None:
				feature_post = encoder.W_enc[:, feature_idx] if not use_decoder else encoder.W_dec[feature_idx]
			bias = -(encoder.b_dec @ feature_post) if use_decoder else encoder.b_enc[feature_idx] - (encoder.b_dec @ feature_post)
			if use_raw_scores:
				cur_scores = (mlp_acts_flattened @ feature_post) + bias
			else:
				_, hidden_acts, _, _, _, _ = encoder(mlp_acts_flattened)
				cur_scores = hidden_acts[:, feature_idx]
			if ignore_endoftext:
					cur_scores[tokens_arr[i:i+batch_size].reshape(-1) == endoftext_token] = -torch.inf
		scores.append(to_numpy(cur_scores.reshape(-1, tokens_arr.shape[1])).astype(np.float16))
	return np.concatenate(scores)

# get indices and values at uniform percentiles of arr
def sample_percentiles(arr, num_samples):
	sample_idxs = []
	sample_vals = []
	
	num_samples = num_samples - 1
	p_step = 100./num_samples
	
	for p in np.arange(0,100,p_step):
		value_at_p = np.percentile(arr.reshape(-1), p, interpolation='nearest')
		p_idx = np.abs(arr-value_at_p).argmin()
		
		sample_vals.append(value_at_p)
		sample_idxs.append(np.unravel_index(p_idx, arr.shape))

	# get maximum
	value_at_p = np.max(arr)
	p_idx = np.abs(arr-value_at_p).argmin()

	sample_vals.append(value_at_p)
	sample_idxs.append(np.unravel_index(p_idx, arr.shape))
	
	return np.array(sample_vals), np.array(sample_idxs)

# get indices and values uniformly spaced throughout arr
def sample_uniform(arr, num_samples, unique=True, use_tqdm=False, only_max_range=False):
    sample_idxs = []
    sample_vals = []

    max_val = np.max(arr)
    if not only_max_range:
        min_val = np.min(arr)
    else:
        min_val = -max_val

    num_samples = num_samples - 1
    p_step = 1./num_samples

    func = (lambda x: x) if not use_tqdm else (lambda x: tqdm.tqdm(x))
    
    for p in func(np.arange(0,1,p_step)):
        value_at_p = min_val + p * (max_val - min_val)
        p_idx = np.abs(arr-value_at_p).argmin()
        real_val = arr[np.unravel_index(p_idx, arr.shape)]
        sample_vals.append(real_val)
        sample_idxs.append(np.unravel_index(p_idx, arr.shape))

    # get maximum
    value_at_p = np.max(arr)
    p_idx = np.abs(arr-value_at_p).argmin()

    sample_vals.append(value_at_p)
    sample_idxs.append(np.unravel_index(p_idx, arr.shape))

    sample_vals = np.array(sample_vals)
    sample_idxs = np.array(sample_idxs)

    if unique:
        sample_vals, sample_pos = np.unique(sample_vals, return_index=True)
        sample_idxs = sample_idxs[sample_pos]
    
    return sample_vals, sample_idxs

import html

def make_sequence_html(token_strs, scores,
    scores_min=None,
    scores_max=None,
    max_color='#ff8c00',
    zero_color='#ffffff',
    return_head=False,
    cur_token_idx=None,
    window_size=None,
):
    if scores_min is None: scores_min = scores.min()
    if scores_max is None: scores_max = scores.max()
    scores_normalized = (scores-scores_min)/(scores_max-scores_min)

    if window_size is not None:
        left_idx = np.max([0, cur_token_idx-window_size])
        right_idx = np.min([len(scores), cur_token_idx+window_size])
        scores = scores[left_idx:right_idx]
        scores_normalized = scores_normalized[left_idx:right_idx]
        token_strs = token_strs[left_idx:right_idx]
        cur_token_idx = cur_token_idx - left_idx

    max_color_vec = np.array([int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)])
    zero_color_vec = np.array([int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)])

    color_vecs = np.einsum('i, j -> ij', scores_normalized, max_color_vec) + np.einsum('i, j -> ij', 1-scores_normalized, zero_color_vec)
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]

    tokens_html = "".join([
        f"""<span class='token'
            style='background-color: {color_strs[i]}'
            onMouseOver='showTooltip(this)'
            onMouseOut='hideTooltip(this)'>{"<b>" if cur_token_idx is not None and i == cur_token_idx else ""}{html.escape(token_str)}{"</b>" if cur_token_idx is not None and i == cur_token_idx else ""}<span class='feature_val'> ({scores[i]:.2f})</span></span>"""
         for i, token_str in enumerate(token_strs)
    ])

    if return_head:
        head = """
<script>
    function showTooltip(element) {
        feature_val = element.querySelector('.feature_val')
        feature_val.style.display='inline'
    }

    function hideTooltip(element) {
        feature_val = element.querySelector('.feature_val')
        feature_val.style.display='none'
    }
</script>
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }

    .feature_val {
        display: none;
        font-family: serif;
    }

    #tooltip {
        display: none;
    }
</style>
"""
        return head + tokens_html
    else:
        return tokens_html

def get_uniform_band_examples(scores, uniform_vals, uniform_idxs, num_bands, band_size, return_percentages=False):
    retlist = []
    
    total_num_exs = num_bands*band_size
    scores_min = scores.min() 
    scores_max = scores.max()
    bandwidth = (scores_max-scores_min)/total_num_exs
    denom = 1 if not return_percentages else np.prod(scores.shape)

    for band in range(0, total_num_exs+band_size, band_size):
        #print(band)
        low_score = scores_min + band*bandwidth
        high_score = scores_min + (band+band_size)*bandwidth
        num_examples_in_band = np.sum(np.logical_and(
            scores > low_score, scores <= high_score
        ))/denom
        retlist.append((low_score, high_score, num_examples_in_band, uniform_idxs[np.logical_and(uniform_vals>=low_score, uniform_vals<=high_score)]))
    return retlist

def display_activating_examples_dash(model, all_tokens, scores,
     num_examples=50,
     num_bands=5,
     bandwidth=10,
     return_percentages=True,
     window_size=5,
     header_level=3
    ):
    if type(header_level) is int:
        header_tag = f'h{header_level}'
    else:
        header_tag = 'p'

    display(HTML(f"<{header_tag} style='font-family: serif'>Firing frequency: {100*np.sum(scores > 0)/np.prod(scores.shape):.4f}%</{header_tag}>"))
    uniform_vals, uniform_idxs = sample_uniform(scores, num_examples, unique=True)
    unif_bands = get_uniform_band_examples(scores, uniform_vals, uniform_idxs, num_bands, bandwidth, return_percentages=return_percentages)
    for band in reversed(unif_bands):
        cur_html_list = [f"<details><summary><{header_tag} style='display: inline; font-family: serif'>Between {band[0]:.2f} and {band[1]:.2f}: {100*band[2]:.4f}%</{header_tag}></summary>"]
        for example_idx, token_idx in reversed(band[3]):
            cur_html_list.append(
                make_sequence_html(
                    model.to_str_tokens(all_tokens[example_idx]), scores[example_idx],
                    scores_min=scores.min(), scores_max=scores.max(), return_head=True, cur_token_idx=token_idx, window_size=window_size
                ) + f"<span> Example {example_idx}, token {token_idx}</span>" + "<br/>"
            )
        cur_html_list.append("</details>")
        display(HTML("".join(cur_html_list)))

def get_logits_for_feature(model, sae, feature_idx, k=7):
    feature = sae.W_dec[feature_idx]

    with torch.no_grad():
        most_pos = torch.topk(feature @ model.W_U, k=k)
        most_neg = torch.topk(-feature @ model.W_U, k=k)
    
    top_vals = to_numpy(most_pos.values)
    top_idxs = to_numpy(most_pos.indices)
    top_tokens = model.to_str_tokens(top_idxs)
    
    bot_vals = to_numpy(-most_neg.values)
    bot_idxs = to_numpy(most_neg.indices)
    bot_tokens = model.to_str_tokens(bot_idxs)

    return zip(top_vals, top_tokens, bot_vals, bot_tokens)
    
def batch_color_interpolate(scores, max_color, zero_color, scores_min=None, scores_max=None):
    if scores_min is None: scores_min = scores.min()
    if scores_max is None: scores_max = scores.max()
    scores_normalized = (scores - scores_min) / (scores_max - scores_min)
    
    max_color_vec = np.array([int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)])
    zero_color_vec = np.array([int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)])

    color_vecs = np.einsum('i, j -> ij', scores_normalized, max_color_vec) + np.einsum('i, j -> ij', 1-scores_normalized, zero_color_vec)
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
    return color_strs

def display_logits_for_feature(model, sae, feature_idx, k=7):
    logits = list(get_logits_for_feature(model, sae, feature_idx, k=k))

    table_html = """
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }
</style>
<table>
    <thead>
        <tr>
            <th colspan=2 style='text-align:center'>Bottom logits</th>
            <th colspan=2 style='text-align:center'>Top logits</th>
        </tr>
    </thead>
    <tbody>
"""

    top_scores = np.array([x[0] for x in logits])
    bot_scores = np.array([x[2] for x in logits])
    scores_max = np.max(top_scores)
    scores_min = np.min(bot_scores)
    
    top_color_strs = batch_color_interpolate(top_scores, '#7f7fff', '#ffffff', scores_min=scores_min, scores_max=scores_max)
    bot_color_strs = batch_color_interpolate(-bot_scores, '#ff7f7f', '#ffffff', scores_min=scores_min, scores_max=scores_max)

    for i, (top_val, top_token, bot_val, bot_token) in enumerate(logits):
        row_html =\
f"""<tr>
    <td style='text-align:left'><span class='token' style='background-color: {bot_color_strs[i]}'>{html.escape(bot_token).replace(' ', '&nbsp;')}</span></td>
    <td style='text-align:right'>{bot_val:.3f}</td>
    <td style='text-align:left'><span class='token' style='background-color: {top_color_strs[i]}'>{html.escape(top_token).replace(' ', '&nbsp;')}</span></td>
    <td style='text-align:right'>+{top_val:.3f}</td>
</tr>"""
        table_html = table_html + row_html
    table_html = table_html + "</tbody></table>"
    display(HTML(table_html))

    all_logits = to_numpy(sae.W_dec[feature_idx] @ model.W_U)

    fig, ax = plt.subplots()
    ax.hist(all_logits[all_logits < 0], color='#ff7f7f')
    ax.hist(all_logits[all_logits > 0], color='#7f7fff')
    fig.set_size_inches(5,2)
    plt.show()
   
def plot_pulledback_feature(model, feature_vector, transcoder, size=None, do_plot=True,
                            input_tokens=None, input_example=None, input_token_idx=None):
    if size is None: size=(5,3)
    with torch.no_grad():
        pulledback_feature = transcoder.W_dec @ feature_vector.vector

        if input_example is not None:
            input_layer = transcoder.cfg.hook_point_layer
            if type(input_example) is int and input_tokens is not None:
                prompt = input_tokens[input_example]
            elif type(input_example) is str:
                prompt = input_example
            # TODO: add list support
            _, cache = model.run_with_cache(prompt, stop_at_layer=input_layer+1,
                names_filter=get_act_name(f'normalized{input_layer}ln2', input_layer)
            )
            feature_activs = transcoder(cache[get_act_name(f'normalized{input_layer}ln2', input_layer)])[1][0,input_token_idx]
            pulledback_feature = pulledback_feature * feature_activs
            
    pulledback_feature = to_numpy(pulledback_feature)

    if do_plot:
        score_max = np.max(np.abs([pulledback_feature.max(), pulledback_feature.min()]))
        score_min = -np.min(np.abs([pulledback_feature.max(), pulledback_feature.min()]))
        colors = batch_color_interpolate(pulledback_feature, '#7f7fff', '#ff7f7f')#, scores_min=score_min, scores_max=score_max)
        
        
        fig, ax = plt.subplots()
        ax.plot(pulledback_feature, alpha=0.5)
        ax.scatter(range(len(pulledback_feature)), pulledback_feature, color=colors)
        fig.set_size_inches(size[0], size[1])
        plt.xlabel("Transcoder feature index")
        plt.ylabel("Connection strength")
        if type(input_example) is int:
            plt.title(f"Example {input_example} token {input_token_idx}:\n {str(feature_vector)} from transcoder features")
        elif type(input_example) is str:
            plt.title(f"Connections on prompt:\n {str(feature_vector)} from transcoder features")
        else:
            plt.title(f"Input-independent connections:\n {str(feature_vector)} from transcoder features")
        plt.show()
    return pulledback_feature

def get_transcoder_pullback_features(model, feature_vector, transcoder, k=7, do_plot=True,
    input_tokens=None, input_example=None, input_token_idx=None
):
    pulledback_feature = plot_pulledback_feature(model, feature_vector, transcoder, do_plot=do_plot,
        input_tokens=input_tokens, input_example=input_example, input_token_idx=input_token_idx)
    pulledback_feature = torch.from_numpy(pulledback_feature)
    with torch.no_grad():
        most_pos = torch.topk(pulledback_feature, k=k)
        most_neg = torch.topk(-pulledback_feature, k=k)
    
    top_vals = to_numpy(most_pos.values)
    top_idxs = to_numpy(most_pos.indices)
    
    bot_vals = to_numpy(-most_neg.values)
    bot_idxs = to_numpy(most_neg.indices)

    return zip(top_vals, top_idxs, bot_vals, bot_idxs)

def display_transcoder_pullback_features(model, feature_vector, transcoder, k=7,
    input_tokens=None, input_example=None, input_token_idx=None
):
    logits = list(get_transcoder_pullback_features(model, feature_vector, transcoder, k=k,
        input_tokens=input_tokens, input_example=input_example, input_token_idx=input_token_idx)
    )

    table_html = """
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }
</style>
<table>
    <thead>
        <tr>
            <th colspan=2 style='text-align:center'>Most-negative transcoder features</th>
            <th colspan=2 style='text-align:center'>Most-positive transcoder features</th>
        </tr>
    </thead>
    <tbody>
"""

    top_scores = np.array([x[0] for x in logits])
    bot_scores = np.array([x[2] for x in logits])
    scores_max = np.max(top_scores)
    scores_min = np.min(bot_scores)
    
    top_color_strs = batch_color_interpolate(top_scores, '#7f7fff', '#ffffff', scores_min=scores_min, scores_max=scores_max)
    bot_color_strs = batch_color_interpolate(-bot_scores, '#ff7f7f', '#ffffff', scores_min=scores_min, scores_max=scores_max)

    for i, (top_val, top_idx, bot_val, bot_idx) in enumerate(logits):
        row_html =\
f"""<tr>
    <td style='text-align:left'><span class='token' style='background-color: {bot_color_strs[i]}'>{bot_idx}</span></td>
    <td style='text-align:right'>{bot_val:.3f}</td>
    <td style='text-align:left'><span class='token' style='background-color: {top_color_strs[i]}'>{top_idx}</span></td>
    <td style='text-align:right'>+{top_val:.3f}</td>
</tr>"""
        table_html = table_html + row_html
    table_html = table_html + "</tbody></table>"
    display(HTML(table_html))

def get_ov_norms_for_transcoder_feature(model, transcoder, feature_idx, layer=None):
    with torch.no_grad():
        propagated_vecs = torch.einsum('lhio,o->lhi', model.OV.AB, transcoder.W_enc[:, feature_idx])
        if layer is not None:
            propagated_vecs = propagated_vecs[:layer+1]
        ov_norms = propagated_vecs.norm(dim=-1)
        
        fig, ax = plt.subplots()
        mat = ax.matshow(to_numpy(ov_norms), cmap='Reds', vmin=0)
        fig.colorbar(mat, location="bottom")
        ax.set_yticks(range(ov_norms.shape[0]))
        ax.set_xlabel("Attention head")
        ax.set_ylabel("Layer")
        fig.set_size_inches(5,3)
        ax.set_title(f"OV de-embedding norms for transcoder feature {feature_idx}", fontsize=10)
        plt.show()

    return ov_norms

def get_deembeddings_for_transcoder_feature(model, transcoder, feature_idx, attn_head=None, attn_layer=0, k=7):
    with torch.no_grad():
        if attn_head is not None:
            pulledback_feature = model.W_E @ model.OV.AB[attn_layer, attn_head] @ transcoder.W_enc[:, feature_idx]
        else:
            pulledback_feature = model.W_E @ transcoder.W_enc[:, feature_idx]
        if k == 0:
            return to_numpy(pulledback_feature)
        else:
            most_pos = torch.topk(pulledback_feature, k=k)
            most_neg = torch.topk(-pulledback_feature, k=k)
    
            top_vals = to_numpy(most_pos.values)
            top_idxs = to_numpy(most_pos.indices)
            top_tokens = model.to_str_tokens(top_idxs)
            
            bot_vals = to_numpy(-most_neg.values)
            bot_idxs = to_numpy(most_neg.indices)
            bot_tokens = model.to_str_tokens(bot_idxs)

            return to_numpy(pulledback_feature), zip(top_vals, top_tokens, bot_vals, bot_tokens)

def get_deembeddings_for_feature_vector(model, feature_vector, k=7):
    with torch.no_grad():
        pulledback_feature = model.W_E @ feature_vector.vector
        if k == 0:
            return to_numpy(pulledback_feature)
        else:
            most_pos = torch.topk(pulledback_feature, k=k)
            most_neg = torch.topk(-pulledback_feature, k=k)
    
            top_vals = to_numpy(most_pos.values)
            top_idxs = to_numpy(most_pos.indices)
            top_tokens = model.to_str_tokens(top_idxs)
            
            bot_vals = to_numpy(-most_neg.values)
            bot_idxs = to_numpy(most_neg.indices)
            bot_tokens = model.to_str_tokens(bot_idxs)

            return to_numpy(pulledback_feature), zip(top_vals, top_tokens, bot_vals, bot_tokens)

def plot_deembedding_for_transcoder_feature(model, transcoder, feature_idx, attn_head=None, attn_layer=0):
    pulledback_feature = get_deembeddings_for_transcoder_feature(model, transcoder, feature_idx, attn_head=attn_head, attn_layer=attn_layer, k=0)
    
    score_max = np.max(np.abs([pulledback_feature.max(), pulledback_feature.min()]))
    score_min = -np.min(np.abs([pulledback_feature.max(), pulledback_feature.min()]))
    colors = batch_color_interpolate(pulledback_feature, '#7f7fff', '#ff7f7f')#, scores_min=score_min, scores_max=score_max)
    
    fig, ax = plt.subplots()
    ax.plot(pulledback_feature, alpha=0.5)
    ax.scatter(range(len(pulledback_feature)), pulledback_feature, color=colors)
    fig.set_size_inches(4,2)
    plt.xlabel("Token index")
    plt.ylabel("Connection strength")
    plt.show()

def display_deembeddings_for_transcoder_feature(model, transcoder, feature_idx, attn_head=None, attn_layer=0, k=7):
    pulledback_feature, deembeddings = get_deembeddings_for_transcoder_feature(model, transcoder, feature_idx, attn_head=attn_head, attn_layer=attn_layer, k=k)
    deembeddings = list(deembeddings)

    table_html = """
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }
</style>"""f"""
<b>{"Direct path" if attn_head is None else f"Attention head {attn_head}"}</b>
<table>
    <thead>
        <tr>
            <th colspan=2 style='text-align:center'>Most-negative de-embedding tokens</th>
            <th colspan=2 style='text-align:center'>Most-positive de-embedding tokens</th>
        </tr>
    </thead>
    <tbody>
"""

    top_scores = np.array([x[0] for x in deembeddings])
    bot_scores = np.array([x[2] for x in deembeddings])
    scores_max = np.max(top_scores)
    scores_min = np.min(bot_scores)
    
    top_color_strs = batch_color_interpolate(top_scores, '#7f7fff', '#ffffff', scores_min=scores_min, scores_max=scores_max)
    bot_color_strs = batch_color_interpolate(-bot_scores, '#ff7f7f', '#ffffff', scores_min=scores_min, scores_max=scores_max)

    for i, (top_val, top_token, bot_val, bot_token) in enumerate(deembeddings):
        row_html =\
f"""<tr>
    <td style='text-align:left'><span class='token' style='background-color: {bot_color_strs[i]}'>{html.escape(bot_token).replace(" ", "&nbsp;")}</span></td>
    <td style='text-align:right'>{bot_val:.3f}</td>
    <td style='text-align:left'><span class='token' style='background-color: {top_color_strs[i]}'>{html.escape(top_token).replace(" ", "&nbsp;")}</span></td>
    <td style='text-align:right'>+{top_val:.3f}</td>
</tr>"""
        table_html = table_html + row_html
    table_html = table_html + "</tbody></table>"
    display(HTML(table_html))

def display_deembeddings_for_feature_vector(model, feature_vector, k=7):
    pulledback_feature, deembeddings = get_deembeddings_for_feature_vector(model, feature_vector, k=k)
    deembeddings = list(deembeddings)

    table_html = """
<style>
    span.token {
        font-family: monospace;
        
        border-style: solid;
        border-width: 1px;
        border-color: #dddddd;
    }
</style>"""f"""
<b>{str(feature_vector)}</b>
<table>
    <thead>
        <tr>
            <th colspan=2 style='text-align:center'>Most-negative de-embedding tokens</th>
            <th colspan=2 style='text-align:center'>Most-positive de-embedding tokens</th>
        </tr>
    </thead>
    <tbody>
"""

    top_scores = np.array([x[0] for x in deembeddings])
    bot_scores = np.array([x[2] for x in deembeddings])
    scores_max = np.max(top_scores)
    scores_min = np.min(bot_scores)
    
    top_color_strs = batch_color_interpolate(top_scores, '#7f7fff', '#ffffff', scores_min=scores_min, scores_max=scores_max)
    bot_color_strs = batch_color_interpolate(-bot_scores, '#ff7f7f', '#ffffff', scores_min=scores_min, scores_max=scores_max)

    for i, (top_val, top_token, bot_val, bot_token) in enumerate(deembeddings):
        row_html =\
f"""<tr>
    <td style='text-align:left'><span class='token' style='background-color: {bot_color_strs[i]}'>{html.escape(bot_token).replace(" ", "&nbsp;")}</span></td>
    <td style='text-align:right'>{bot_val:.3f}</td>
    <td style='text-align:left'><span class='token' style='background-color: {top_color_strs[i]}'>{html.escape(top_token).replace(" ", "&nbsp;")}</span></td>
    <td style='text-align:right'>+{top_val:.3f}</td>
</tr>"""
        table_html = table_html + row_html
    table_html = table_html + "</tbody></table>"
    display(HTML(table_html))

def display_analysis_for_transcoder_feature(model, transcoder, feature_idx, attn_k=2, k=8, layer=None):
    display(HTML(f"<h3>Transcoder feature {feature_idx}</h3>"))
    display_deembeddings_for_transcoder_feature(model, transcoder, feature_idx, k=k)
    plot_deembedding_for_transcoder_feature(model, transcoder, feature_idx)
    
    display(HTML(f"<h4>OV circuits</h4>"))
    ov_norms = get_ov_norms_for_transcoder_feature(model, transcoder, feature_idx, layer=layer)
    top_ov_norms, top_ov_heads_flattened = torch.topk(ov_norms.flatten(), k=attn_k)
    top_ov_indices = np.array(np.unravel_index(to_numpy(top_ov_heads_flattened), ov_norms.shape)).T        

    colors = batch_color_interpolate(to_numpy(top_ov_norms), '#ff7f7f', '#ffffff', scores_min=0, scores_max=top_ov_norms.max().item())
    table_html = f"""<h4>Top {top_ov_indices.shape[0]} attention heads by OV circuit norm</h4>
    <table>
        <thead>
            <tr>
                <th style='text-align:center'>Layer</th>
                <th style='text-align:center'>Head</th>
                <th style='text-align:center'>Norm</th>
            </tr>
        </thead>
        <tbody>
            { "".join(f"<tr style='background-color: {color}'><td>{layer}</td><td>{head}</td><td>{norm:.2f}</td>" for (layer, head), norm, color in zip(top_ov_indices, top_ov_norms, colors))}
        </tbody>
    """
    display(HTML(table_html))

    for cur_layer, head in top_ov_indices:
        display_deembeddings_for_transcoder_feature(model, transcoder, feature_idx, attn_head=head, attn_layer=cur_layer)
        plot_deembedding_for_transcoder_feature(model, transcoder, feature_idx, attn_head=head, attn_layer=cur_layer)