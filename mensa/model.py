import torch
import torch.nn.functional as F
import dgl
import tqdm

class SAGE(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layer = num_layer
        self.layers.append(dgl.nn.SAGEConv(in_size, hid_size, "mean"))
        self.bns.append(torch.nn.BatchNorm1d(hid_size))
        for _ in range(num_layer - 2):
            self.layers.append(dgl.nn.SAGEConv(hid_size, hid_size, "mean"))
            self.bns.append(torch.nn.BatchNorm1d(hid_size))
        self.layers.append(dgl.nn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = torch.nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    def new_forward(self, blocks, x, fixed_layer):
        h = x
        self.eval()
        with torch.no_grad():
            for l in range(fixed_layer):
                h = self.layers[l](blocks[l], h)
                if l != len(self.layers) - 1:
                    h = self.bns[l](h)
                    h = F.relu(h)
        self.train()
        for l in range(fixed_layer, len(self.layers)):
            h = self.layers[l](blocks[l], h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    
    def first_k_forward(self, g, x, fixed_layer):
        self.eval()
        with torch.no_grad():
            for l in range(fixed_layer):
                x = self.layers[l](g, x)
                if l != len(self.layers) - 1:
                    x = self.bns[l](x)
                    x = F.relu(x)
        return x
    
    def last_k_forward(self, blocks, x, fixed_layer):
        self.train()
        h = x
        for l in range(fixed_layer, len(self.layers)):
            layer, block = self.layers[l], blocks[l - fixed_layer]
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    def fullbatch_inference(self, g, device):
        h = g.ndata["feat"].to(device)
        g = g.to(device)
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes].to(device)
                h = layer(blocks[0].to(device), x)
                if l != len(self.layers) - 1:
                    h = self.bns[l](h)
                    h = F.relu(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        
        return y