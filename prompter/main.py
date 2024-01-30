import wandb
import argparse

from utils import *
from mmengine.config import Config
from dataset import DataFolder
from criterion import build_criterion
from models.dpa_p2pnet import build_model
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser('Cell prompter')
    parser.add_argument('--config', default='pannuke123.py', type=str)
    parser.add_argument('--run-name', default=None, type=str, help='wandb run name')
    parser.add_argument('--group-name', default=None, type=str, help='wandb group name')

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Run Mode
    parser.add_argument('--eval', action='store_true')

    # * Train
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--print-freq", default=5, type=int, help="print frequency")
    parser.add_argument("--use-wandb", action='store_true', help='use wandb for logging')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs.')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs.')
    parser.add_argument('--clip-grad', type=float, default=0.1,
                        help='Clip gradient norm (default: 0.1)')
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )

    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=1,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99)",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # * Distributed training
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()

    return opt


def main():
    args = parse_args()
    init_distributed_mode(args)
    set_seed(args)

    cfg = Config.fromfile(f'config/{args.config}')
    if args.output_dir:
        mkdir(f'checkpoint/{args.output_dir}')
        cfg.dump(f'checkpoint/{args.output_dir}/config.py')

    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    model_without_ddp = model

    train_dataset = DataFolder(cfg, 'train')
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn
    )

    try:
        val_dataset = DataFolder(cfg, 'val')
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            drop_last=False
        )
    except FileNotFoundError:
        pass

    test_dataset = DataFolder(cfg, 'test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=False
    )

    if args.eval:
        # checkpoint = torch.load(f'./checkpoint/{args.resume}/best.pth', map_location="cpu")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get('model_ema', ckpt['model']))
        evaluate(
            cfg,
            model,
            test_dataloader,
            device,
            calc_map=True
        )
        return

    model_ema = None
    if args.model_ema:
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=args.model_ema_decay)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = build_criterion(cfg, device)
    actual_lr = cfg.optimizer.lr * (cfg.data.batch_size_per_gpu * get_world_size()) / 8  # linear scaling rule
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_without_ddp.parameters()),
        lr=actual_lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    scaler = torch.cuda.amp.Gradcaler() if args.amp else None

    if args.use_wandb and is_main_process():
        wandb.init(
            project='Prompter',
            name=args.run_name,
            group=args.group_name,
            config=vars(args),
        )

    # load checkpoint
    max_cls_f1 = 0
    if args.resume:
        # checkpoint = torch.load(f'./checkpoint/{args.resume}/latest.pth', map_location="cpu")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        max_cls_f1 = checkpoint.get("f1", 0)
        if model_ema:
            model_ema.module.load_state_dict(checkpoint["model_ema"])
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        log_info = train_one_epoch(
            args,
            model,
            train_dataloader,
            criterion,
            optimizer,
            epoch,
            device,
            model_ema,
            scaler
        )

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "f1": max_cls_f1,
                "epoch": epoch,
                "args": args
            }

            if model_ema:
                checkpoint["model_ema"] = model_ema.module.state_dict()

            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()

            save_on_master(
                checkpoint,
                f"checkpoint/{args.output_dir}/latest.pth",
            )

        try:
            if epoch >= args.start_eval:
                metrics, metrics_string = evaluate(
                    cfg,
                    model_ema or model,
                    val_dataloader,
                    device,
                    epoch,
                )

                log_info.update(dict(zip(["Det Pre", "Det Rec", "Det F1"], metrics['Det'])))
                log_info.update(dict(zip(["Cls Pre", "Cls Rec", "Cls F1"], metrics['Cls'])))
                log_info.update(dict(IoU=metrics['IoU']))

                cls_f1 = metrics['Cls'][-1]
                if max_cls_f1 < cls_f1:
                    max_cls_f1 = cls_f1

                    checkpoint = {
                        "model": model_without_ddp.state_dict() if not model_ema else model_ema.module.state_dict(),
                        "metrics": metrics_string,
                        "f1": max_cls_f1,
                        "epoch": epoch,
                    }
                    if args.output_dir:
                        save_on_master(
                            checkpoint,
                            f"checkpoint/{args.output_dir}/best.pth",
                        )
        except NameError:
            pass

        if is_main_process() and args.use_wandb:
            wandb.log(
                log_info,
                step=epoch
            )

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
