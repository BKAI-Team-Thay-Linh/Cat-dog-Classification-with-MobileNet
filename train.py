import os 
import argparse

from src.model import DogCatModel
from src.dataset import DogCatDataset

import torch 
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m',type=str,default='resnet',
                        help='model name')
    parser.add_argument('--max_epochs','-me',type=int,default=20,
                        help='max epoch')
    parser.add_argument('--batch_size','-bs',type=int,default=64,
                        help='batch size')
    parser.add_argument('--lr','-l',type=float,default=2e-4,
                        help='learning rate')
    parser.add_argument('--num_workers','-nw',type=int,default=0,
                        help='number of workers')
    parser.add_argument('--seed','-s',type=int,default=42,
                        help='seed')
    parser.add_argument('--wandb','-w',default=False,action='store_true',
                        help='use wandb or not')
    parser.add_argument('--wandb_key','-wk',type=str,
                        help='wandb API key')
    args = parser.parse_args()
    
    # SEED
    pl.seed_everything(args.seed,workers=True)

    # WANDB (OPTIONAL)
    if args.wandb:
        wandb.login(key = args.wandb_key) # API KEY
        name = f"{args.model}-{args.max_epochs}-{args.batch_size}-{args.lr}"
        logger  = WandbLogger(project="cat-dog-cnn-architectures",
                            name=name,
                            log_model="all")
    else:
        logger = None
    
    # DATALOADER 
    train_dataset = DogCatDataset(
        root_dir=os.path.join(os.getcwd(),'data'),
        type='train'
    )
    
    train_dataset, val_dataset = random_split(
        dataset=train_dataset,
        lengths=(0.8,0.2),
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_dataset, test_dataset = random_split(
        dataset=train_dataset,
        lengths=(0.8,0.2),
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )
    
    # MODEL
    model = DogCatModel(model=args.model, lr=args.lr)
    
    # CALLBACK
    root_path = os.path.join(os.getcwd(),"checkpoints")
    ckpt_path = os.path.join(os.path.join(root_path,f"{args.model}/"))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_callback = ModelCheckpoint(
        monitor    = "val_acc",
        dirpath    = ckpt_path,
        filename   = "checkpoints-{epoch:02d}-{val_acc:.5f}",
        save_top_k = 1,
        mode       = "max"
    ) # save top 3 epochs with the lowest validation loss
    lr_callback    = LearningRateMonitor("step")

    
    # TRAINER
    trainer = pl.Trainer(default_root_dir    = root_path,
                         logger              = logger,
                         callbacks           = [ckpt_callback,lr_callback],
                         gradient_clip_val   = 0.5,
                         max_epochs          = args.max_epochs,
                         enable_progress_bar = True,
                         deterministic       = True)
    
    # FIT MODEL
    trainer.fit(model             = model,
                train_dataloaders = train_loader,
                val_dataloaders   = val_loader)
    
    # TEST MODEL
    trainer.test(dataloaders = test_loader)

if __name__ == '__main__':
    main()