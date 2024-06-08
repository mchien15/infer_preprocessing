# Preprocessing Module
### Skip PDF Images, Unwarping, Small Rotation, and Document Page Detection

## How to Use

### Install Required Packages

This project is developed with Python 3.8. To get started, install the necessary packages by running:

```
pip install -r requirements.txt
```

### Preprocessing multiple images in a folder
<!-- 
#### Option 1: Use `inference.ipynb` file -->

#### You can run the script with the example images provided by executing:
```
python src/page_dewarp/inference.py
```
Output images will be stored in the `example_output` folder

#### To test the preprocessing on your own images, use the following command:
```
python src/page_dewarp/inference.py --input_path=/path/to/your/input/folder --output_path=/path/to/your/output/folder
```