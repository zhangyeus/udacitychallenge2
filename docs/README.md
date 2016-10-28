
1.train and test
python main.py --dataset /data/output --test-dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --test-batch_size 80
2.evaluation model
python evaluation_model.py --dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80

3.evaluation  model with existed cnn_weights.hdf5

python evaluation_model_load_weights.py --dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --weights-path=/data/cnn_weights.hdf5




train data path:/home/wei/Documents/challenge2/data/output/dataset/

test data path:/home/wei/Documents/challenge2/finalpre/output/dataset-2-2/

hdf5: /home/wei/Documents/challenge2/mymodel/self-driving/src/cnn_weights.hdf5

1.
python main.py --dataset /home/wei/Documents/challenge2/data/output/dataset --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --test-batch_size 80

python main.py --dataset /home/wei/Documents/challenge2/data/output/dataset --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --test-batch_size 16


2.
python evaluation_model.py --dataset /home/wei/Documents/challenge2/data/output/dataset/  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80

python evaluation_model.py --dataset /home/wei/Documents/challenge2/data/output/dataset/  --nb-epoch 2 --resized-image-width 120 --resized-image-height 160


3.
python evaluation_model_load_weights.py --dataset /home/wei/Documents/challenge2/data/output/dataset/  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --weights-path=/home/wei/Documents/challenge2/mymodel/self-driving/src/cnn_weights.hdf5
