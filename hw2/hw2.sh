# TODO: create shell script for running the testing code of the baseline model
WGET='https://www.dropbox.com/s/yprj7vkhhfu5i2e/model_best.pth.tar?dl=1'
DIR='model_baseline.pth.tar'
wget -O $DIR $WGET 
python3 hw2.py --data_dir $1 --output_dir $2 
