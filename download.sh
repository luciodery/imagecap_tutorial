wget https://www.dropbox.com/s/lmwbhnjrxbvoo01/Data.zip
unzip Data.zip
rm Data.zip

mkdir models
wget https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip -P ./models/
unzip ./models/pretrained_model.zip -d ./models/
rm ./models/pretrained_model.zip
