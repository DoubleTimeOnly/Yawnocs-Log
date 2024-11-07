from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from yawnoc.deep_solver.dataset import ConwayDataset
from yawnoc.deep_solver.solver_module import LitSolver
from yawnoc.deep_solver.utils import get_dataloader_from_cfg, get_hydra_cfg


def train(cfg: OmegaConf):
    """
    Train a network to predict a previous generation given a current board state

    Args:
        cfg (OmegaConf):
            Configuration data structure containing experiment and model params
    """
    # instantiate objects from hydra configs
    train_dataloader = get_dataloader_from_cfg(cfg.train)
    test_dataloader = get_dataloader_from_cfg(cfg.test)
    save_dir = r"C:\Users\Victor\Documents\Projects\conway_game\experiment_results"
    loggers = [
        CSVLogger(save_dir=save_dir, name=cfg.experiment_name),
        TensorBoardLogger(save_dir=save_dir, name=cfg.experiment_name, default_hp_metric=False)
    ]
    model = hydra.utils.instantiate(cfg.lit_model, _convert_="partial")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)
    
    # save configuration with rest of the experiment
    log_dir = Path(loggers[0].log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    with open(log_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    trainer.fit(model=model, train_dataloaders=train_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)


def test():
    """
    Quick way to test a trained model
    """
    board_size = (10, 10)
    test_dataset = ConwayDataset(board_size=board_size, length=128)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32, 
    )
    model = LitSolver.load_from_checkpoint(
        r"C:\Users\Victor\Documents\Projects\conway_game\experiment_results\test\version_26\checkpoints\epoch=999-step=1000.ckpt",
    )
    trainer = L.Trainer()
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    cfg = get_hydra_cfg(
        config="train_estimator.yaml",
        overrides=[],
    )
    cfg.board_size = (5, 5)
    cfg.batch_size = 64
    cfg.lit_model.inter_features = 256
    cfg.lit_model.series_length = (2, 10)
    cfg.trainer.max_steps = 10000
    train(cfg=cfg)
    # test()
