# test f1 score

python small_graph.py --dataset flickr --layers 5 --epochs 50 --hid 256 --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.1
python small_graph.py --dataset ogbn-arxiv --layers 5 --epochs 50 --hid 256 --partsize 10000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5
python small_graph.py --dataset reddit --layers 5 --epochs 50 --hid 256 --partsize 10000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5

python large_graph.py --dataset yelp --layers 4 --epochs 50 --hid 512  --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.1
python large_graph.py --dataset amazon --layers 4 --epochs 50 --hid 512 --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5
python large_graph.py --dataset ogbn-products --layers 4 --epochs 50 --hid 512 --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5

# memory usage

#python small_graph.py --dataset flickr --layers 5 --epochs 50 --hid 256 --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.1 --mem_bench
#python small_graph.py --dataset ogbn-arxiv --layers 5 --epochs 50 --hid 256 --partsize 10000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5 --mem_bench
#python small_graph.py --dataset reddit --layers 5 --epochs 50 --hid 256 --partsize 10000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5 --mem_bench

#python large_graph.py --dataset yelp --layers 4 --epochs 50 --hid 512  --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.1 --mem_bench
#python large_graph.py --dataset amazon --layers 4 --epochs 50 --hid 512 --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5 --mem_bench
#python large_graph.py --dataset ogbn-products --layers 4 --epochs 50 --hid 512 --partsize 5000 --batch_size 5 --lr 0.005 --wd 0 --dropout 0.5 --mem_bench


