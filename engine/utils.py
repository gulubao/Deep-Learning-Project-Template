def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1

    if iter % log_period == 0:
        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                    .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))


def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['ce_loss']
    logger.info("Training Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                .format(engine.state.epoch, avg_accuracy, avg_loss))

if val_loader is not None:
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['ce_loss']
        logger.info("Validation Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss)
                    )

# adding handlers using `trainer.on` decorator API
def print_times(engine):
    logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                .format(engine.state.epoch, timer.value() * timer.step_count,
                        train_loader.batch_size / timer.value()))
    timer.reset()