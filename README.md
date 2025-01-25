# Scalable GNN Training via Parameter Freeze and Layer Detachment
Accepted by DASFAA 2025
### How to Run
For dataset Flickr, Reddit, Yelp and Amazon, download them from [Google Drive link](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) or [BaiduYun link(code:f1ao)](https://pan.baidu.com/share/init?surl=SOb0SiSAXavwAcNqkttwcg) and put them at correct place. Only four files are needed: adj_full.npz, class_map.json, feats.npy and role.json. These public datasets are collected by GraphSAINT and are irrelevant to this paper.

```
/PATH/TO/THIS_DIR
|   
└───mensa
| 
└───dataset/
|   |   flickr/
|   |   reddit/
|   |   ...
```

For Ogbn-arxiv and Ogbn-products, they will be downloaded automatically.

If you want to run algorithm mensa:
```shell
cd mensa
bash run.sh
```

Results will be saved at log directory.
