import torch
import dgl
from sklearn.metrics import f1_score
import numpy as np

def compute_acc(logits, labels):
    if (len(labels.shape) == 1):
        pred = logits.detach().argmax(1)
    else:
        pred = logits.detach()
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

    pred = pred.cpu()
    labels = labels.cpu()
    return f1_score(labels, pred, average='micro')   

def fullbatch_infer(device, graph, nid, model):
    model.eval()
    with torch.no_grad():
        pred = model.fullbatch_inference(graph, device)
    label = graph.ndata["label"]
    if (len(nid) == 2):
        valid_nid, test_nid = nid
        valid_pred = pred[valid_nid]
        test_pred = pred[test_nid]
        valid_label = label[valid_nid]
        test_label = label[test_nid]
        return compute_acc(valid_pred, valid_label), compute_acc(test_pred, test_label)
    else:
        test_pred = pred[nid[0]]
        test_label = label[nid[0]]
        return compute_acc(test_pred, test_label)    

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )
        if (len(nid) == 2):
            valid_nid, test_nid = nid
            valid_pred = pred[valid_nid]
            test_pred = pred[test_nid]
            valid_label = graph.ndata["label"][valid_nid]
            test_label = graph.ndata["label"][test_nid]
            return compute_acc(valid_pred, valid_label), compute_acc(test_pred, test_label)
        else:
            test_pred = pred[nid[0]]
            test_label = graph.ndata["label"][nid[0]]
            return compute_acc(test_pred, test_label)
        
        
def sampled_infer(device, graph, nid, model, batch_size): # only for valid nid
    model.eval()
    with torch.no_grad():
        sampler = dgl.dataloading.NeighborSampler([5] * model.num_layer)
        eval_dataloader = dgl.dataloading.DataLoader(
            graph.cpu(),
            nid.cpu(),
            sampler,
            device=torch.device('cpu'),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        pred, label = [], []
        for it, (input_nodes, output_nodes, blocks) in enumerate(eval_dataloader):
            blocks = [blk.to(device) for blk in blocks]
            x = blocks[0].srcdata['feat'].to(device)
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            pred.append(y_hat.cpu())
            label.append(y.cpu())
        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)
        return compute_acc(pred, label)

def check_computation_graph_size(g, device, args):
    train_idx = torch.where(g.ndata['train_mask'] == 1)[0]
    
    sampler = dgl.dataloading.NeighborSampler(
        [args.fanout] * args.layers
    )

    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    tot_block_size = []
    
    for it, (_, _, blocks) in enumerate(train_dataloader):
        tot = []
        s = 0
        # for i, blk in enumerate(blocks):
        #     if (i == 0):
        #         tot.append(blk.srcdata[dgl.NID])
        #     else:
        #         tot.append(blk.dstdata[dgl.NID])
        # tot = torch.unique(torch.cat(tot, dim=0))
        for i, blk in enumerate(blocks):
            if (i == 0):
                s += blk.num_src_nodes()
            else:
                s += blk.num_dst_nodes()
        tot_block_size.append(s)
        
    return int(np.mean(tot_block_size))  