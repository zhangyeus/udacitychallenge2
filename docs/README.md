
1.train and test
python main.py --dataset /data/output --test-dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --test-batch_size 80
2.evaluation model
python evaluation_model.py --dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80

3.evaluation  model with existed cnn_weights.hdf5

python evaluation_model_load_weights.py --dataset /data/output  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --weights-path=/data/cnn_weights.hdf5

4.load weights and predict
python original-predict.py --test-dataset /data/output --weights-path /data/cnn_weights.hdf5


5.final(2 training dataset)
python main.py --dataset1 /data/output --dataset2 /data/output --test-dataset /data/output

6.load weight
python weightloader.py --dataset1 /data/output --dataset2 /data/output --dataset3 /data/output --test-dataset /data/output --weights_path1 /data/cnn_weights.hdf5

7. finally using training weights to predict
python final-predict.py --test-dataset /data/output --weights-path /data/cnn_weights.hdf5

8. training + new 21 folders
python trainNew.py --dataset1 /data/output --dataset2 /data/output --dataset3 /data/output --test-dataset /data/output

################################################################################


train data path1:/home/ubuntu/dataset/929/output/dataset/
train data path2:/home/ubuntu/dataset/1003/output/dataset-2-2/
train data path3:/home/ubuntu/dataset/testdata/testdata/Challenge2/Train/

test data path:/home/ubuntu/dataset/testdata/testdata/Challenge2/Test/

weightpath: /home/ubuntu/code/testcode/successfulModel/core_constraint30_1024by2/cnn_weights.hdf5

1.
python main.py --dataset /home/wei/Documents/challenge2/data/output/dataset --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --test-batch_size 80

python main.py --dataset /home/wei/Documents/challenge2/data/output/dataset --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2  --nb-epoch 20 --resized-image-width 60 --resized-image-height 80 --test-batch_size 80


2.
python evaluation_model.py --dataset /home/wei/Documents/challenge2/data/output/dataset/  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80

python evaluation_model.py --dataset /home/wei/Documents/challenge2/data/output/dataset/  --nb-epoch 2 --resized-image-width 120 --resized-image-height 160


3.
python evaluation_model_load_weights.py --dataset /home/wei/Documents/challenge2/data/output/dataset/  --nb-epoch 2 --resized-image-width 60 --resized-image-height 80 --weights-path=/home/wei/Documents/challenge2/successful_weights/core_constraint/cnn_weights.hdf5


4.
python original-predict.py --test-dataset /home/ubuntu/dataset/929/output/dataset/ --weights-path /home/ubuntu/code/testcode/successfulModel/core_constraint30_1024by2/cnn_weights.hdf5

python original-predict.py --test-dataset /home/ubuntu/dataset/1003/output/dataset-2-2/ --weights-path /home/ubuntu/code/testcode/successfulModel/core_constraint30_1024by2/cnn_weights.hdf5




5.
python main.py --dataset1 /home/wei/Documents/challenge2/finalpre/output/dataset-2-2/ --dataset2 /home/wei/Documents/challenge2/data/output/dataset/ --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2/

python main.py --dataset1 /home/wei/Documents/challenge2/data/output/dataset/ --dataset2 /home/wei/Documents/challenge2/finalpre/output/dataset-2-2/ --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2/


6.6.load weight
python weightloader.py --dataset1 /home/ubuntu/dataset/929/output/dataset/ --dataset2 /home/ubuntu/dataset/1003/output/dataset-2-2/ --dataset3 /home/ubuntu/dataset/testdata/testdata/Challenge2/Train/ --test-dataset /home/wei/Documents/challenge2/finalpre/output/dataset-2-2/ --weights_path1 /home/ubuntu/code/testcode/successfulModel/core_constraint30_1024by2/cnn_weights.hdf5

7.
python final-predict.py --test-dataset /home/ubuntu/dataset/testdata/testdata/Challenge2/Test/ --weights-path /home/ubuntu/code/testcode/successfulModel/core_constraint30_1024by2/cnn_weights.hdf5

8.
python trainNew.py --dataset1 /home/ubuntu/dataset/929/output/dataset/ --dataset2 /home/ubuntu/dataset/1003/output/dataset-2-2/ --dataset3 /home/ubuntu/dataset/testdata/testdata/Challenge2/Train/ --test-dataset /data/output

