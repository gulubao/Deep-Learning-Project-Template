# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

import logging
from .inference import evaluate
from .utils import log_training_loss, log_training_results, print_times

def do_train(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    log_period = args.SOLVER.LOG_PERIOD
    checkpoint_period = args.SOLVER.CHECKPOINT_PERIOD
    output_dir = args.OUTPUT_DIR
    device = args.MODEL.DEVICE
    epochs = args.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")

    n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    start_time = time.time()
    for epoch in range(self.last_epoch + 1, args.epoches):
        if dist.is_dist_available_and_initialized():
            self.train_dataloader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
            args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

        self.lr_scheduler.step()
        
        if self.output_dir:
            checkpoint_paths = [self.output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.checkpoint_step == 0:
                checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
        )

        # TODO 
        for k in test_stats.keys():
            if k in best_stat:
                best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                best_stat[k] = max(best_stat[k], test_stats[k][0])
            else:
                best_stat['epoch'] = epoch
                best_stat[k] = test_stats[k][0]
        print('best_stat: ', best_stat)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if self.output_dir and dist.is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (self.output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    """https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch"""
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print("targets, in def train_one_epoch\n", targets)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
