# small graph: flickr, ogbn-arxiv, reddit

import argparse
import dgl
import torch
import torch.nn.functional as F
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
)
import time
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from datasets import YelpDataset, AmazonDataset
import numpy as np
from logger import *
from utils import *
from model import *

def train(device, g, model, multilabel, args, partition_graphs):
    train_idx = torch.where(g.ndata['train_mask'] == 1)[0]
    val_idx = torch.where(g.ndata['val_mask'] == 1)[0]
    
    sampler = NeighborSampler(
        [args.fanout1] * args.layers
    )
    
    train_dataloader = DataLoader(
        g,
        train_idx.to(device),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    
    new_sampler = NeighborSampler(
        [args.fanout2] * (args.layers - args.fixed),
    )
    
    new_train_dataloader = DataLoader(
        g,
        train_idx.to(device),
        new_sampler, 
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    

    opt1 = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd)
    opt2 = torch.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd)


    best_valid_acc = 0
    train_time = 0   
    tot_iteration = 0
    
    z = torch.zeros((g.num_nodes(), args.hid)).to(device)
    
    for epoch in range(args.epochs):
        model.train()
        t1 = time.time()

        for it, (_, _, blocks) in enumerate(train_dataloader):

            blocks = [blk.to(device) for blk in blocks]
            x = blocks[0].srcdata["feat"].to(device)
            y = blocks[-1].dstdata["label"].to(device)
            y_hat = model(blocks, x)
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat), reduction="mean")
            else:
                loss = F.cross_entropy(y_hat, y)
            opt1.zero_grad()
            loss.backward()
            opt1.step()
            tot_iteration += 1
            
        for sg in partition_graphs:
            sg = sg.to(device)
            id = sg.ndata[dgl.NID]
            x = g.ndata['feat'][id].to(device)
            with torch.no_grad():
                x = model.first_k_forward(sg, x, args.fixed)
            z[id] = x
            
            
        for it, (input_node, _, new_blocks) in enumerate(new_train_dataloader):

            new_blocks = [blk.to(device) for blk in new_blocks]
            input_node = input_node.to(device)
            x = z[input_node]
            y = new_blocks[-1].dstdata["label"].to(device)
            y_hat = model.last_k_forward(new_blocks, x, fixed_layer=args.fixed)
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat), reduction="mean")
            else:
                loss = F.cross_entropy(y_hat, y)
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            tot_iteration += 1
        
        t2 = time.time()
        train_time += t2 - t1
        
        PRINT(f'{epoch}th epoch time: {t2 - t1:.2f}s')

        if (epoch + 1) % 2 == 0:
            valid_acc = sampled_infer(device, g, val_idx, model, batch_size=512)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), f'./pt/{name}.pt')
            PRINT(
                "Epoch {:05d} | Accuracy {:.4f} | Best Valid Acc {:.4f}".format(
                    epoch, valid_acc, best_valid_acc
                )
            )
            
    z = z.cpu()
        
    PRINT(f'train time: {train_time:.2f}s')                                   

    return train_time


def mem_bench(device, g, model, multilabel, args):
    max_mem = 0
    torch.cuda.empty_cache()
    
    train_idx = torch.where(g.ndata['train_mask'] == 1)[0]
    
    sampler = NeighborSampler(
        [args.fanout1] * args.layers
    )
    
    train_dataloader = DataLoader(
        g.cpu(),
        train_idx.cpu(),
        sampler,
        device=torch.device('cpu'),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    new_sampler = NeighborSampler(
        [args.fanout2] * (args.layers - args.fixed),
    )
    
    new_train_dataloader = DataLoader(
        g.cpu(),
        train_idx.cpu(),
        new_sampler, 
        device=torch.device('cpu'),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    
    opt1 = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd)
    opt2 = torch.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd)
    
    z = torch.zeros((g.num_nodes(), args.hid))
    
    for it, (_, _, blocks) in enumerate(train_dataloader):
        if (it > len(train_dataloader) * 0.5):
            break 
        blocks = [blk.to(device) for blk in blocks]
        x = blocks[0].srcdata["feat"].to(device)
        y = blocks[-1].dstdata["label"].to(device)
        y_hat = model(blocks, x)
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat), reduction="mean")
        else:
            loss = F.cross_entropy(y_hat, y)
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        
        peak_usage = torch.cuda.max_memory_allocated(device)
        max_mem = max(max_mem, peak_usage)
        torch.cuda.empty_cache()
        
    model = model.to('cpu')
    
    for sg in partition_graphs:
        id = sg.ndata[dgl.NID]
        x = g.ndata['feat'][id]
        with torch.no_grad():
            x = model.first_k_forward(sg, x, args.fixed)
        z[id] = x
        
    
    model = model.to(device)
    torch.cuda.empty_cache()
        
    for it, (input_node, _, new_blocks) in enumerate(new_train_dataloader):
        if (it > len(new_train_dataloader) * 0.5):
            break

        new_blocks = [blk.to(device) for blk in new_blocks]
        x = z[input_node].to(device)
        y = new_blocks[-1].dstdata["label"].to(device)
    
        y_hat = model.last_k_forward(new_blocks, x, fixed_layer=args.fixed)
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat), reduction="mean")
        else:
            loss = F.cross_entropy(y_hat, y)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        
        peak_usage = torch.cuda.max_memory_allocated(device)
        max_mem = max(max_mem, peak_usage)
        torch.cuda.empty_cache()
        
    PRINT(f'cuda peak usage: {max_mem / (1024 ** 2):.2f}MB')
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--fixed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fanout1", type=int, default=2)
    parser.add_argument("--fanout2", type=int, default=5)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--lr1", type=float, default=0.01)
    parser.add_argument("--lr2", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mem_bench", action="store_true")
    parser.add_argument("--part", type=int, default=10)
    args = parser.parse_args()
    
    log_path = "./log"
    f = new_log(log_path, args)
    def PRINT(text):
        return PRINT_LOG(text, file=f)


    # load and preprocess dataset
    PRINT("Loading data")
    
    dataset_root = '../dataset'
    if args.dataset == "yelp":
        name = "yelp"
        dataset = YelpDataset(raw_dir=dataset_root)
    elif args.dataset == "amazon":
        name = "amazon"
        dataset = AmazonDataset(raw_dir=dataset_root)
    elif args.dataset == "ogbn-products":
        name = "ogbn-products"
        dataset = AsNodePredDataset(DglNodePropPredDataset(name, root=dataset_root))
    else:
        raise ValueError("dataset should be [yelp, amazon, ogbn-products]")
    
    PRINT(f'dataset: {name}')

    g = dataset[0]
    multilabel = len(g.ndata['label'].shape) > 1
    g = dgl.to_bidirected(g, copy_ndata=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    partition_graphs = dgl.metis_partition(g, args.part)
    partition_graphs = [sg for i, sg in partition_graphs.items()]
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
            train_time = train(device, g, model, multilabel, args, partition_graphs)
            tot_time.append(train_time)

            # test the model
            PRINT("Testing...")
            model.load_state_dict(torch.load(f'./pt/{name}.pt'))
            model = model.to(device)
            test_idx = torch.where(g.ndata['test_mask'] == 1)[0]
            acc = layerwise_infer(device, g, [test_idx], model, batch_size=4096)
            PRINT("Test Accuracy {:.4f}".format(acc.item()))
            tot_acc.append(acc)
            
        PRINT(f'average acc: {np.mean(tot_acc) * 100:.2f} ± {np.std(tot_acc) * 100:.2f}%')
        PRINT(f'average train time: {np.mean(tot_time):.2f} ± {np.std(tot_time):.2f}s')
    
        
        