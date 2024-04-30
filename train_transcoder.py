# Transcoder training sample code

"""
This sample script can be used to train a transcoder on a model of your choice.
This code, along with the transcoder training code more generally, was largely
    adapted from an older version of Joseph Bloom's SAE training repo, the latest
    version of which can be found at https://github.com/jbloomAus/SAELens.
Most of the parameters given here are the same as the SAE training parameters
    listed at https://jbloomaus.github.io/SAELens/training_saes/.
Transcoder-specific parameters are marked as such in comments.

"""

import torch
import os 
import sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model

lr = 0.0004 # learning rate
l1_coeff = 0.00014 # l1 sparsity regularization coefficient

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)

    # "hook_point" is the TransformerLens HookPoint representing
    #    the input activations to the transcoder that we want to train on.
    # Here, "ln2.hook_normalized" refers to the activations after the
    #    pre-MLP LayerNorm -- that is, the inputs to the MLP.
    # You might alternatively prefer to train on "blocks.8.hook_resid_mid",
    #    which corresponds to the input to the pre-MLP LayerNorm.
    hook_point = "blocks.8.ln2.hook_normalized",
    hook_point_layer = 8,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    model_name='gpt2-small',

    # Transcoder-specific parameters.
    is_transcoder = True, # We're training a transcoder here.
    # "out_hook_point" is the TransformerLens HookPoint representing
    #    the output activations that the transcoder should reconstruct.
    # In our use case, we're using transcoders to interpret MLP sublayers.
    # This means that our transcoder will take in the input to an MLP and
    #    attempt to spit out the output of the MLP (but in the form of a
    #    sparse linear combination of feature vectors).
    # As such, we want to grab the "hook_mlp_out" activations from our
    #    transformer, which (as the name suggests), represent the
    #    output activations of the original MLP sublayer.
    out_hook_point = "blocks.8.hook_mlp_out",
    out_hook_point_layer = 8,
    d_out = 768,
    
    # SAE Parameters
    expansion_factor = 32,
    b_dec_init_method = "mean",
    
    # Training Parameters
    lr = lr,
    l1_coefficient = l1_coeff,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    total_training_tokens = 1_000_000 * 60,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method = None,
    feature_sampling_window = 1000,
    resample_batches=1028,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-8,

    # WANDB
    log_to_wandb = False,
    
    # Misc
    use_tqdm = True,
    device = "cuda",
    seed = 42,
    n_checkpoints = 3,
    checkpoint_path = "gpt2-small-transcoders", # change as you please
    dtype = torch.float32,
)

print(f"About to start training with lr {lr} and l1 {l1_coeff}")
print(f"Checkpoint path: {cfg.checkpoint_path}")
print(cfg)

loader = LMSparseAutoencoderSessionloader(cfg)
model, sparse_autoencoder, activations_loader = loader.load_session()

# train SAE
sparse_autoencoder = train_sae_on_language_model(
    model, sparse_autoencoder, activations_loader,
    n_checkpoints=cfg.n_checkpoints,
    batch_size = cfg.train_batch_size,
    feature_sampling_method = cfg.feature_sampling_method,
    feature_sampling_window = cfg.feature_sampling_window,
    feature_reinit_scale = cfg.feature_reinit_scale,
    dead_feature_threshold = cfg.dead_feature_threshold,
    dead_feature_window=cfg.dead_feature_window,
    use_wandb = cfg.log_to_wandb,
    wandb_log_frequency = cfg.wandb_log_frequency
)

# save sae to checkpoints folder
path = f"{cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
sparse_autoencoder.save_model(path)