from config.structured import DataloaderConfig, ProjectConfig
from .custom_dataset import Landmarker
from torch.utils.data.dataloader import DataLoader

def get_dataset(cfg: ProjectConfig):
    if cfg.dataset.type == 'landmark':
        dataloader_cfg: DataloaderConfig = cfg.dataloader
        train_dataset = Landmarker('train')
        val_dataset = Landmarker('val')
        dataloader_train = DataLoader(train_dataset, batch_size=dataloader_cfg.batch_size,
                                      num_workers=dataloader_cfg.num_workers, shuffle=True, drop_last=True,
                                      persistent_workers=dataloader_cfg.num_workers > 0)
        dataloader_val = dataloader_vis = DataLoader(val_dataset, batch_size=dataloader_cfg.batch_size,
                                                     num_workers=dataloader_cfg.num_workers, shuffle=False,
                                                     drop_last=False)
    else:
        raise NotImplementedError(cfg.dataset.type)

    return dataloader_train, dataloader_val, dataloader_vis
