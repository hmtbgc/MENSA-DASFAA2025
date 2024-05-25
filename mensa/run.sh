# test f1 score

python small_graph.py --dataset flickr --layers 5 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 256 --lr1 0.01 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5
python small_graph.py --dataset ogbn-arxiv --layers 5 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 256 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5
python small_graph.py --dataset reddit --layers 5 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 256 --lr1 0.01 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5

python large_graph.py --dataset yelp --layers 4 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 512 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.2
python large_graph.py --dataset amazon --layers 4 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 512 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --part 20
python large_graph.py --dataset ogbn-products --layers 4 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 512 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --part 30



# memory usage

# python small_graph.py --dataset flickr --layers 5 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 256 --lr1 0.01 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench 
# python small_graph.py --dataset ogbn-arxiv --layers 5 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 256 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench 
# python small_graph.py --dataset reddit --layers 5 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 256 --lr1 0.01 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench 

# python large_graph.py --dataset yelp --layers 4 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 512 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.2 --mem_bench 
# python large_graph.py --dataset amazon --layers 4 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 512 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench --part 20
# python large_graph.py --dataset ogbn-products --layers 4 --fixed 2 --epochs 50 --fanout1 2 --fanout2 5 --hid 512 --lr1 0.005 --lr2 0.005 --wd 0 --batch_size 2048 --dropout 0.5 --mem_bench --part 30