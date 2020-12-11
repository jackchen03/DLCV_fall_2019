# TODO: create shell script for running the testing code of your improved model
WGET='https://www.dropbox.com/s/qm21tmmuwrn4cgt/model_best.pth.tar?dl=1'
DIR='model_improved.pth.tar'
wget  -O $DIR $WGET 
python3 hw2_best.py --data_dir $1 --output_dir $2 
