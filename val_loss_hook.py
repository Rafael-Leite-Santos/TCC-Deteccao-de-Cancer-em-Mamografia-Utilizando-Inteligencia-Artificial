import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ValLossHook(Hook):
    """Calcula a loss nas imagens de validação ao fim de cada época
    e a entrega ao EarlyStoppingHook como 'val_loss'."""

    priority = 'NORMAL'  # roda antes do EarlyStoppingHook e do LoggerHook

    def __init__(self, dataloader):
        self.dataloader_cfg = dataloader
        self.loader = None

    def before_train(self, runner):
        self.loader = runner.build_dataloader(self.dataloader_cfg)

    def after_val_epoch(self, runner, metrics=None):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for data in self.loader:
                data = model.data_preprocessor(data, True)
                losses = model(**data, mode='loss')
                loss, _ = model.parse_losses(losses)  # soma das 4 losses
                total += loss.item()
                n += 1
        val_loss = total / max(n, 1)
        runner.message_hub.update_scalar('val/val_loss', val_loss)  # aparece no log
        if metrics is not None:
            metrics['val_loss'] = val_loss  # visto pelo EarlyStoppingHook
        runner.logger.info(f'Loss de validacao: {val_loss:.4f}')