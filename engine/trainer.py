# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""
import math
import torch
import os
import time
from .inference import evaluate
from .utils import unwrap_model, AverageMeter

def do_train(
        args,
        start_epoch,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        tb_writer
):

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    for epoch in range(start_epoch, args.max_epochs):
        args.logger.info(f'Start epoch {epoch}')
        train_one_epoch(model, train_loader, loss_fn, epoch, optimizer, scheduler, args, tb_writer=tb_writer)
        completed_epoch = epoch + 1

        # evaluate
        if (completed_epoch == args.max_epochs or ((completed_epoch % args.evaluate_period) == 0) or completed_epoch==1) and (val_loader is not None):
            evaluate(model, train_loader, completed_epoch, args, tb_writer=tb_writer, data_type = "train")
            evaluate(model, val_loader, completed_epoch, args, tb_writer=tb_writer, data_type = "test")

        # Saving checkpoints.
        if completed_epoch == args.max_epochs or ((completed_epoch % args.checkpoint_period) == 0) or completed_epoch==1:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint_dict, args.checkpoint_path / f"epoch_{completed_epoch}.pt")
            # Count the number of files matching args.checkpoint_path / f"epoch_{completed_epoch}.pt".
            checkpoint_files = list(args.checkpoint_path.glob("epoch_*.pt"))
            # If the number of matching files exceeds 3, delete the one with the smallest completed_epoch until there are exactly 3 files.
            if len(checkpoint_files) > 3:
                sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
                for file_to_delete in sorted_checkpoints[:-3]:
                    file_to_delete.unlink()


def train_one_epoch(model, dataloader, loss, epoch, optimizer, scheduler, args, tb_writer=None):
    """open_clip/src/training/train.py --> def train_one_epoch"""
    device = torch.device(args.device)
    model.train()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        if scheduler is not None:
            scheduler(step)
        images, texts, idxs = batch
        images = images.to(device=device, non_blocking=args.non_blocking)
        texts = texts.to(device=device, non_blocking=args.non_blocking)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if args.accum_freq == 1:
            model_out = model(images, texts)
            logit_scale = model_out["logit_scale"]
            losses = loss(**model_out, output_dict=True)
            total_loss = sum(losses.values())
            losses["loss"] = total_loss
            total_loss.backward()
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                model_out = model(images, texts)
                for f in ("logit_scale", "logit_bias"):
                    model_out.pop(f, None)
                for key, val in model_out.items():
                    if key in accum_features:
                        accum_features[key].append(val)
                    else:
                        accum_features[key] = [val]
                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                model_out = model(images, texts)

                inputs_no_accum = {}
                inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                if "logit_bias" in model_out:
                    inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                inputs = {}
                for key, val in accum_features.items():
                    accumulated = accum_features[key]
                    inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                del inputs
                del inputs_no_accum
                total_loss = sum(losses.values())
                losses["loss"] = total_loss
                total_loss.backward()

        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            args.logger.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for