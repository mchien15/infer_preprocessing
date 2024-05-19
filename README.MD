# Preprocessing Module (skip pdf images + unwarping + small rotation)

## How to use
### Install required packages
```
pip install -r requirements.txt
```
### Preprocessing multiple images in a folder
<!-- 
#### Option 1: Use `inference.ipynb` file -->

#### Run `inference.py` file with example images:
```
python src/page_dewarp/inference.py
```
Output images will be stored in `example_output` folder

#### Or you can test with your own images by using:
```
python src/page_dewarp/inference.py --input_path=/path/to/your/input/folder --output_path=/path/to/your/output/folder
```