mkdir data
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
# wget http://images.cocodataset.org/zips/train2014.zip -P ./data/
wget http://images.cocodataset.org/zips/val2014.zip -P ./data/


mkdir models
wget https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip -P ./models/
unzip ./models/pretrained_model.zip -d ./models/
rm ./models/pretrained_model.zip

unzip ./data/captions_train-val2014.zip -d ./data/
rm ./data/captions_train-val2014.zip

# unzip ./data/train2014.zip -d ./data/
# rm ./data/train2014.zip

unzip ./data/val2014.zip -d ./data/
rm ./data/val2014.zip
