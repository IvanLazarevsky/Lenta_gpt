from tqdm import tqdm_notebook
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def train_general(seq2seq_model,
                  optimizer,
                  compute_training_loss,
                  compute_val_loss,
                  train_loader,
                  val_loader,
                  epochs,
                  n_prints,
                  gradient_accumulation_steps=1,
                  after_epoch=None,
                  after_gradient=None,
                  after_step=None,
                  total_steps=0,
                  initial_epoch=0,
                  device='cuda'):
    context = AttrDict({'model': seq2seq_model, 'optimizer': optimizer, 'device': device, 'terminate': False})
    for epoch in tqdm_notebook(range(initial_epoch, initial_epoch + epochs)):
        seq2seq_model.train()
        epoch_state = AttrDict({'epoch': epoch})

        total_batches = len(train_loader)

        divide_at_last_batch = total_batches % gradient_accumulation_steps == 0

        print_every = total_batches // n_prints
        running_loss = 0.0
        batch_tqdm_iter = tqdm_notebook(enumerate(train_loader), total=total_batches)

        for iteration, batch in batch_tqdm_iter:
            total_steps += 1
            #             if iteration % gradient_accumulation_steps == 0:
            #                 optimizer.zero_grad()
            batch_state = AttrDict()

            batch_state.batch = batch
            batch_state.epoch = epoch
            batch_state.iteration = iteration
            batch_state.total_steps = total_steps

            loss = compute_training_loss(context, batch_state)
            loss_to_display = loss.item()

            if iteration + 1 != total_batches or divide_at_last_batch:
                loss = loss / gradient_accumulation_steps

            if context.terminate:
                return

            running_loss += loss.item()

            batch_state.loss = loss.item()

            if iteration % print_every == print_every - 1:
                print(f'Epoch {epoch+1} Iteration {iteration+1} Loss {running_loss / (iteration+1)}')

            loss.backward()

            if (iteration + 1) % gradient_accumulation_steps == 0 or (iteration + 1) == total_batches:
                if after_gradient:
                    after_gradient(context, batch_state)
                    if context.terminate:
                        return
                optimizer.step()
                optimizer.zero_grad()
                if after_step:
                    after_step(context, batch_state)
                    if context.terminate:
                        return

            current_lr = next(iter(optimizer.param_groups))['lr']
            batch_tqdm_iter.set_postfix({'loss': loss_to_display, 'lr': current_lr})

        seq2seq_model.eval()

        with torch.no_grad():
            val_loss = 0.0
            for batch in tqdm_notebook(val_loader, total=len(val_loader)):
                batch_state = AttrDict()
                batch_state.batch = batch
                batch_state.epoch = epoch

                loss = compute_val_loss(context, batch_state)
                if context.terminate:
                    return
                val_loss += loss.item()

            epoch_state.val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch+1} val_Loss {epoch_state.val_loss}')

        if after_epoch:
            after_epoch(context, epoch_state)
            if context.terminate:
                return