n_epochs = 20
She raised her eyebrows with a hint of


output_dir  = 'llama_debug_model'
os.makedirs(output_dir, exist_ok=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)


train_losses, val_losses, track_tokens_seen = [], [], []
tokens_seen = 0
global_step = -1
start_time = time.time()


# try:
print("Training ...")
for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Generate text passage
            if global_step % print_sample_iter == 0:
                generate_and_print_sample(
                    model, tokenizer, device, start_context
                )

        if global_step % save_ckpt_freq:
            file_name = output_dir / f"model_pg_{global_step}.pth"
            torch.save(model.state_dict(), file_name)
            print(f"Saved {file_name}")

        print_eta(start_time, book_start_time, index, total_files)

# except KeyboardInterrupt:
#     file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
#     torch.save(model.state_dict(), file_name)
#     print(f"Saved {file_name}")