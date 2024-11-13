# code from https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb

import argparse
import math
import os
from pathlib import Path
import time


# modified. tokenizer import
# import tiktoken
from transformers import PreTrainedTokenizerFast
import torch
from previous_chapters import (
    # create_dataloader_v1, # modified. use create_dataloader_v2 instead
    create_dataloader_v2,
    GPTModel,
    generate_and_print_sample,
    calc_loss_batch,
    evaluate_model,
    plot_losses
)


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data



# def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
#     split_idx = int(train_ratio * len(text_data))
#     train_loader = create_dataloader_v1(
#         text_data[:split_idx],
#         batch_size=batch_size,
#         max_length=max_length,
#         stride=stride,
#         drop_last=True,
#         shuffle=True,
#         num_workers=num_workers
#     )
#     val_loader = create_dataloader_v1(
#         text_data[split_idx:],
#         batch_size=batch_size,
#         max_length=max_length,
#         stride=stride,
#         drop_last=False,
#         shuffle=False,
#         num_workers=num_workers
#     )
#     return train_loader, val_loader


def create_dataloaders(train_ratio, batch_size, num_workers=0):
    ''' 
    modified.
    parameter: text_data is removed
    parameter: max_length, stride are removed
    '''
    train_loader, val_loader = create_dataloader_v2(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        train_ratio=train_ratio
    )
    return train_loader, val_loader


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    #  "Estimated Time of Arrival" or "Estimated Time of Completion".
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)
    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")

BOOK_VERSION = True


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, output_dir, tokenizer,
                warmup_steps,resume_global_step=None, initial_lr=3e-05, min_lr=1e-6):
    print("Training ...")
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    try:
        for epoch in range(n_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                global_step += 1
                # modified to resume
                if resume_global_step and global_step < resume_global_step:
                    print('.')
                    continue    # continue till global_step gets to resume_global_step

                # Adjust the learning rate based on the current phase (warmup or cosine annealing)
                if global_step < warmup_steps:
                    # Linear warmup
                    lr = initial_lr + global_step * lr_increment  
                else:
                    # Cosine annealing after warmup
                    progress = ((global_step - warmup_steps) / 
                                (total_training_steps - warmup_steps))
                    lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

                # Apply the calculated learning rate to the optimizer
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(lr)  # Store the current learning rate

                # Calculate and backpropagate the loss
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()

                # Apply gradient clipping after the warmup phase to avoid exploding gradients

                if BOOK_VERSION:
                    if global_step > warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                else:
                    if global_step >= warmup_steps:  # the book originally used global_step > warmup_steps, which lead to a skipped clipping step after warmup
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                optimizer.step()
                tokens_seen += input_batch.numel()

                # Periodically evaluate the model on the training and validation sets
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader,
                        device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    # Print the current losses
                    print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                    )

            # Generate and print a sample from the model to monitor progress (at the end of each epoch)
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )
    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        # modified. to save optimizer state_dict along with model state dict
        # torch.save(model.state_dict(), file_name)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            }, 
            file_name
        )
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen, track_lrs

def get_max_global_step_file(directory='model_checkpoints'):
    """
    Finds the filename with the maximum global step in a given directory.

    Args:
        directory: The directory containing the files.

    Returns:
        The filename with the maximum global step.
    """

    max_step = 0
    max_file = None

    for filename in os.listdir(directory):
        if filename.startswith('model_pg_') and filename.endswith('.pth'):
            global_step = int(filename.split('_')[2])
            if global_step > max_step:
                max_step = global_step
                max_file = filename

    if max_file:return os.path.join(directory, max_file), max_step
    return None, None

if __name__ == "__main__":
    # Note:
    # Uncomment the following code to calculate the execution time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--data_dir', type=str, default='gutenberg/data',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')
    parser.add_argument('--max_text_len', type=int, default=45000000,
                        help='testing different text sizes.')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='whether or not to load the previously saved model if it exists')

    args = parser.parse_args()

    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": 50000,     # Vocabulary size
            "context_length": 10,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50000,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    torch.manual_seed(123)
    
    
    model = GPTModel(GPT_CONFIG_124M)
    peak_lr = 0.001  # this was originally set to 5e-4 in the book by mistake
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)  # the book accidentally omitted the lr assignment

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step=0
    if args.load_model and os.path.exists(output_dir):
        checkpoint_file_path, global_step = get_max_global_step_file(output_dir)
        if checkpoint_file_path:
            print(f'Loading existing model: {os.path.join(output_dir, "model_pg_final.pth")}')
            
            checkpoint = torch.load(checkpoint_file_path, weights_only=True)
            
            # modified (added model loading code)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.to(device)

    

    # modified. code to load the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")

    # modified
    # n_epochs = 15
    n_epochs = args.n_epochs

    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    for index, file_path in enumerate(all_files, 1):
        book_start_time = time.time()
        text_data = read_text_file(file_path) + " <|endoftext|> "
        text_data = text_data[:args.max_text_len]
        print(f"Tokenizing file {index} of {total_files}: {file_path}")

        # Initialize new data loaders for each book
        train_loader, val_loader = create_dataloaders(
            text_data,
            train_ratio=0.9,
            batch_size=args.batch_size,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            num_workers=0
        )
        
        total_steps = len(train_loader) * n_epochs
        warmup_steps = int(0.2 * total_steps) # 20% warmup
        print(warmup_steps)
        
        train_losses, val_losses, tokens_seen, lrs = train_model(
            model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
            eval_freq=100, eval_iter=1, start_context="रामले भात", # "Every effort moves you", <modified>
            output_dir=output_dir, tokenizer=tokenizer, warmup_steps=warmup_steps, resume_global_step=global_step,
            initial_lr=1e-5, min_lr=1e-5
        )
        epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

        print_eta(start_time, book_start_time, index, total_files)

        
        # modified. to save optimizer state_dict along with model state dict
        # torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_losses": train_losses,

            }, 
            output_dir / "model_pg_final.pth"
        )
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Note:
    # Uncomment the following code to show the execution time
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")