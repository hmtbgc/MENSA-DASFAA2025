# test f1 score

python small_graph.py --dataset flickr --layers 5 --epochs 50 --fanout 5 --hid 256 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5
python small_graph.py --dataset ogbn-arxiv --layers 5 --epochs 50 --fanout 5 --hid 256 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 
python small_graph.py --dataset reddit --layers 5 --epochs 50 --fanout 5 --hid 256 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 

python large_graph.py --dataset yelp --layers 4 --epochs 50 --fanout 5 --hid 512 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.1 
python large_graph.py --dataset amazon --layers 4 --epochs 50 --fanout 5 --hid 512 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 
python large_graph.py --dataset ogbn-products --layers 4 --epochs 50 --fanout 5 --hid 512 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 


# memory usage 

# python small_graph.py --dataset flickr --layers 5 --epochs 50 --fanout 5 --hid 256 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench
# python small_graph.py --dataset ogbn-arxiv --layers 5 --epochs 50 --fanout 5 --hid 256 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench
# python small_graph.py --dataset reddit --layers 5 --epochs 50 --fanout 5 --hid 256 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench

# python large_graph.py --dataset yelp --layers 4 --epochs 50 --fanout 5 --hid 512 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.1 --mem_bench
# python large_graph.py --dataset amazon --layers 4 --epochs 50 --fanout 5 --hid 512 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench
# python large_graph.py --dataset ogbn-products --layers 4 --epochs 50 --fanout 5 --hid 512 --lr 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench

