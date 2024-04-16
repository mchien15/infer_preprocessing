# image_folder="/home/asus/Downloads/data_preprocess"
# image_folder='/home/asus/page-dewarp/test-split'
image_folder='/home/asus/stuDYING/IT/Thesis/from_server/pdf1_image'

# Iterate over each image in the folder
for image_path in "$image_folder"/*; do
    # Check if the file is a regular file (i.e., not a directory)
    if [ -f "$image_path" ]; then
        # Run the page-dewarp command on the current image
        python ../src/page_dewarp/__main__.py -d 3 -x 10 -y 50 -el 15 -ea 3 "$image_path"
    fi
done