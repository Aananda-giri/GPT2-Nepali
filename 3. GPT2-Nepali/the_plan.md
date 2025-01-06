
* Reference: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg
Design Decisions and Improvements
Note that this code focuses on keeping things simple and minimal for educational purposes. The code could be improved in the following ways to improve modeling performance and training efficiency:

Modify the prepare_dataset.py script to strip the Gutenberg boilerplate text from each book file.
Update the data preparation and loading utilities to pre-tokenize the dataset and save it in a tokenized form so that it doesn't have to be re-tokenized each time when calling the pretraining script.
Update the train_model_simple script by adding the features introduced in Appendix D: Adding Bells and Whistles to the Training Loop, namely, cosine decay, linear warmup, and gradient clipping.
Update the pretraining script to save the optimizer state (see section 5.4 Loading and saving weights in PyTorch in chapter 5; ch05.ipynb) and add the option to load an existing model and optimizer checkpoint and continue training if the training run was interrupted.
Add a more advanced logger (for example, Weights and Biases) to view the loss and validation curves live
Add distributed data parallelism (DDP) and train the model on multiple GPUs (see section A.9.3 Training with multiple GPUs in appendix A; DDP-script.py).
Swap the from scratch MultiheadAttention class in the previous_chapter.py script with the efficient MHAPyTorchScaledDotProduct class implemented in the Efficient Multi-Head Attention Implementations bonus section, which uses Flash Attention via PyTorch's nn.functional.scaled_dot_product_attention function.
Speeding up the training by optimizing the model via torch.compile (model = torch.compile) or thunder (model = thunder.jit(model)).
Implement Gradient Low-Rank Projection (GaLore) to further speed up the pretraining process. This can be achieved by just replacing the AdamW optimizer with the provided GaLoreAdamW provided in the GaLore Python library.