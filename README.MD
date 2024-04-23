# Preprocessing Module
## How to run
### Install required packages
```
pip install -r requirements.txt
```
### For unwarping a single image
Make a folder to contain output image, for example, `out_file`, then navigate to that folder
```
mkdir out_file && cd out_file
```
Run this command
```
python ../src/page_dewarp/temp_image.py /path/to/input/image
```
### For unwarping multiple images in a folder
Run `inference.ipynb` file