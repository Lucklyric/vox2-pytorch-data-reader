# vox2-pytorch-data-reader

## Required Library
pydub
torch
torchvision
pandas
numpy
imageio
scikit-iamge
glob
librosa

## Download small-scale dataset from the google-drive

## API Usage 
```python
# Change PATHs to you local PATHs
CSV_META = "./vox2/vox2_meta_small.csv" # "test with modified small scale meta csv"
VIDEO_PREFIX = "./vox2/vox2_dev_mp4/dev/mp4/" # "change to your own local path"
AUDIO_PREFIX = "./vox2/vox2_aac/dev/aac/" # "change to your own local path"
BATCH_SIZE = 4
db = DataReader(
            csv_meta = CSV_META,
            audio_prefix = AUDIO_PREFIX,
            video_prefix= VIDEO_PREFIX,
            random = True,
            engine = "librosa"
            )
    
print("==================================")
print("Test signle output:")
test_single_input = db[0]
```
Sample usage can check "__main__" function in "vox2_data_reader.py" sample code. 
   
