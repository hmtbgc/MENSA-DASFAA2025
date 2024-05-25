# small graph: flickr, ogbn-arxiv, reddit

import argparse
import dgl
import torch
import torch.nn.functional as F
import time, math, random
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from datasets import RedditDataset, FlickrDataset
import numpy as np
from logger import *
from utils import *
from model import *

def train(device, g, model, multilabel, args):
    val_idx = torch.where(g.ndata['val_mask'] == 1)[0]
    test_idx = torch.where(g.ndata['test_mask'] == 1)[0]
    
    num_partitions = math.ceil(g.num_nodes() / args.partsize)
    
    sampler = dgl.dataloading.ClusterGCNSampler(
            g,
            num_partitions,
        )
        
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(num_partitions),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_valid_acc = 0
    best_test_acc = 0
    train_time = 0   
    tot_iteration = 0
    
    
    for epoch in range(args.epochs):
        model.train()
        t1 = time.time()
        
        for j, subg in enumerate(dataloader):
            subg = subg.to(device)
            x = subg.ndata['feat'].to(device)
            y = subg.ndata['label'].to(device)
            m = subg.ndata["train_mask"].bool().to(device)
            y_hat = model(subg, x)
                
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat[m], y[m].type_as(y_hat), reduction="mean")
            else:
                loss = F.cross_entropy(y_hat[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_iteration += 1
            
        t2 = time.time()
        train_time += t2 - t1
        
        PRINT(f'{epoch}th epoch time: {t2 - t1:.2f}s')

        if (epoch + 1) % 2 == 0:
            valid_acc, test_acc = fullbatch_infer(device, g.to(device), [val_idx, test_idx], model)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                torch.save(model.state_dict(), f'./pt/{name}.pt')
            PRINT(
                "Epoch {:05d} | Accuracy {:.4f} | Best Valid Acc {:.4f} | Best Test Acc {:.4f}".format(
                    epoch, valid_acc, best_valid_acc, best_test_acc
                )
            )
        
    PRINT(f'average train time: {train_time:.2f}s')

    return train_time


def mem_bench(device, g, model, multilabel, args):
    max_mem = 0
    torch.cuda.empty_cache()
    
    num_partitions = math.ceil(g.num_nodes() / args.partsize)
    
    sampler = dgl.dataloading.ClusterGCNSampler(
            g,
            num_partitions,
        )
        
    dataloader = dgl.dataloading.DataLoader(
        g.cpu(),
        torch.arange(num_partitions).cpu(),
        sampler,
        device=torch.device('cpu'),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for j, subg in enumerate(dataloader):
        subg = subg.to(device)
        x = subg.ndata['feat'].to(device)
        y = subg.ndata['label'].to(device)
        m = subg.ndata["train_mask"].bool().to(device)
        y_hat = model(subg, x)
            
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(y_hat[m], y[m].type_as(y_hat), reduction="mean")
        else:
            loss = F.cross_entropy(y_hat[m], y[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        peak_usage = torch.cuda.max_memory_allocated(device)
        max_mem = max(max_mem, peak_usage)
        torch.cuda.empty_cache()
    PRINT(f'cuda peak usage: {max_mem / (1024 ** 2):.2f}MB')
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--partsize", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mem_bench", action="store_true")
    args = parser.parse_args()
    
    log_path = "./log"
    f = new_log(log_path, args)
    def PRINT(text):
        return PRINT_LOG(text, file=f)

    if os.path.exists('./cluster_gcn.pkl'):
        os.remove('./cluster_gcn.pkl')

    # load and preprocess dataset
    PRINT("Loading data")
    
    dataset_root = '../dataset'
    if args.dataset == "flickr":
        name = "flickr"
        dataset = FlickrDataset(raw_dir=dataset_root)
    elif args.dataset == "reddit":
        name = "reddit"
        dataset = RedditDataset(raw_dir=dataset_root)
    elif args.dataset == "ogbn-arxiv":
        name = "ogbn-arxiv"
        dataset = AsNodePredDataset(DglNodePropPredDataset(name, root=dataset_root))
    else:
        raise ValueError("dataset should be [flickr, reddit, ogbn-arxiv]")
    
    PRINT(f'dataset: {name}')

    g = dataset[0]
    multilabel = len(g.ndata['label'].shape) > 1
    g = dgl.to_bidirected(g, copy_ndata=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = dataset.num_classes
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    
    tot_acc = []
    tot_time = []
    tot_throughput = []
    
    if (args.mem_bench):
        model = SAGE(in_size, args.hid, out_size, args.layers, args.dropout).to(device)
        mem_bench(device, g, model, multilabel, args)
        
    else:
        
        for i in range(5):
            model = SAGE(in_size, args.hid, out_size, args.layers, args.dropout).to(device)
            
            PRINT("Training...")
            train_time = train(device, g, model, multilabel, args)
            tot_time.append(train_time)

            # test the model
            PRINT("Testing...")
            model.load_state_dict(torch.load(f'./pt/{name}.pt'))
            model = model.to(device)
            test_idx = torch.where(g.ndata['test_mask'] == 1)[0]
            acc = fullbatch_infer(device, g.to(device), [test_idx], model)
            PRINT("Test Accuracy {:.4f}".format(acc.item()))
            tot_acc.append(acc)
            
        PRINT(f'average acc: {np.mean(tot_acc) * 100:.2f} ± {np.std(tot_acc) * 100:.2f}%')
        PRINT(f'average train time: {np.mean(tot_time):.2f} ± {np.std(tot_time):.2f}s')
        
        