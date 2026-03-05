import torch
import torch.nn as nn
import logging
from tqdm import tqdm

from .eval import evaluate
from .utils import *

def train(
        args,model,
        train_loader,test_loader,
        model_name=None,
        early_stop = True,
        best = False,
    ):

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "lambda" not in n and p.requires_grad], 
            "lr": args.learning_rate, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False
    )

    # prepare training
    criterion = nn.CrossEntropyLoss()
    device = args.device
    model.to(device)
    model.zero_grad()
    global_setp = 0
    if args.num_steps>0:
        num_steps = args.num_steps
    else:
        num_steps = args.num_epochs*len(train_loader)
    
    best_acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
    last_acc = best_acc
    cnt = 0
    wandb_log(global_setp,accuracy=best_acc)
    if model_name is None:
        model_name = f"{name2abb[args.model_name]}({args.task_name}-{args.pruning_rate:.1f}-FT).pt"
    print("\n=================================================================================")
    logger = logging.getLogger(__name__)
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # start training
    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader, 
            desc="Training (X / X Steps) (loss=X.X)", 
            bar_format="{l_bar}{r_bar}", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            # forward+backward
            if args.task_type == "recognition":
                xb, yb = batch[:2]
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
            elif args.task_type == "detection":
                data = model.data_preprocessor(batch, True)
                losses = model._run_forward(data, mode='loss')
                loss = sum(losses['loss_rpn_cls']) + sum(losses['loss_rpn_bbox']) + losses['loss_cls'] + losses['loss_bbox']
            elif args.task_type == "language":
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_setp += 1
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.4f) (best_acc=%.2f)" % (
                    global_setp, num_steps, loss.item(), best_acc
                )
            )
            # log the training process
            wandb_log(
                global_setp,loss=loss.item(),
            )
            # evaluate
            if global_setp % args.eval_every == 0:
                acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
                wandb_log(global_setp,accuracy=acc)
                if acc > best_acc:
                    best_acc = acc
                save_model(args,model,model_name)
                '''============ Early Stop ============='''
                if abs(acc-last_acc)<1.0:
                    cnt+=1
                    last_acc = acc
                else:
                    cnt = 0
                    last_acc = acc
                if cnt>=3 and early_stop:
                    return acc
                '''====================================='''

            if global_setp >= num_steps: break
        # evaluate
        acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
        wandb_log(global_setp,accuracy=acc)
        if acc > best_acc:
            best_acc = acc
        save_model(args,model,model_name)
        '''============ Early Stop ============='''
        if abs(acc-last_acc)<1.0:
            cnt+=1
            last_acc = acc
        else:
            cnt = 0
            last_acc = acc
        if cnt>=3 and early_stop:
            return acc
        '''====================================='''
        if global_setp >= num_steps: break
    
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("=================================================================================\n")

    return best_acc if best else acc