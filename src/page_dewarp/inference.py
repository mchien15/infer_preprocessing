import onnxruntime as ort
import cv2
import numpy as np
import os
import time
import argparse
from utils import *
from skimage.transform import rotate
from temp_image import main as unwarping_module
from ultralytics import YOLO

pdf_photo_model = ort.InferenceSession('src/page_dewarp/model/pdf_photo.onnx')
pdf_photo_model_input_name, pdf_photo_model_output_name = pdf_photo_model.get_inputs()[0].name, pdf_photo_model.get_outputs()[0].name

sar_model = ort.InferenceSession('src/page_dewarp/model/sar.onnx')
sar_model_input_name, sar_model_output_name = sar_model.get_inputs()[0].name, sar_model.get_outputs()[0].name

page_detect_model = YOLO('src/page_dewarp/model/best.onnx')

def main(input_path, output_path, cleanup):

    if not os.path.exists(output_path):
        print('Creating output directory: ' + output_path)
        os.makedirs(output_path)

    for image in sorted(os.listdir(input_path)):
        try:
            if image == '.gitkeep':
                continue

            start_time = time.time()

            img_path = os.path.join(input_path, image)

            print('Processing image: ' + img_path)

            img = cv2.imread(img_path)

            classes = ['pdf', 'photo']

            # plt.imshow(img)
            # plt.show()
            resized_img = cv2.resize(img, (480, 480))
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            result = pdf_photo_model.run([pdf_photo_model_output_name], {pdf_photo_model_input_name: np.array([resized_img], dtype=np.float32)})

            # print(result)
            print('Classified as class: ' + classes[np.argmax(result)])

            end_time = time.time()

            pdf_photo_time = end_time - start_time

            print('Classification time: ', pdf_photo_time)


            if classes[np.argmax(result)] == 'pdf':
                cv2.imwrite(os.path.join(output_path, image), img)
                print('-' * 50)
                continue
            else:

                start_time = time.time()

                results = page_detect_model.predict(img, save_crop=False)

                box = results[0].boxes.cpu().xyxy[0].numpy()
                x1, y1, x2, y2 = map(int, box)

                cropped_img = img[y1:y2, x1:x2]

                end_time = time.time()

                print('Detection time: ', end_time - start_time)

                start_time = time.time()
                # print(img_path)
                pil_image = unwarping_module(img_path, actual_image=cropped_img)
                # print(type(pil_image))
                unwarped_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                end_time = time.time()

                unwarping_time = end_time - start_time

                print('Unwarping time: ', unwarping_time)

                start_time = time.time()

                rs_img = resizeAndPad(unwarped_img, (480, 480))
                rs_img = cv2.cvtColor(rs_img, cv2.COLOR_BGR2RGB)
                rs_img = rs_img / 255.0
                # print(rs_img.shape)
                result = sar_model.run([sar_model_output_name], {sar_model_input_name: np.array([rs_img], dtype=np.float32)})

                rotated = rotate_small_angle(unwarped_img, result[0])

                # gray_img = rgb2gray(img)
                # resized_image = resize(img, (img.shape[0] // 4, img.shape[1] // 4))

                # rotated = rotate(img, get_skew_angle(resized_image), cval=1)

                # print('Rotated by: ', result[0], ' degrees')

                rotate_time = time.time() - start_time

                print('Rotatingtime: ', rotate_time)

                results = page_detect_model.predict(rotated)

                box = results[0].boxes.cpu().xyxy[0].numpy()
                x1, y1, x2, y2 = map(int, box)

                cropped_img = rotated[y1:y2, x1:x2]

                resized_img = cv2.resize(cropped_img, (cropped_img.shape[1]//4, cropped_img.shape[0]//4))

                gray_image = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

                start_time = time.time()

                skew_angle = skew_angle_hough_transform_avarage(gray_image)

                rotated = rotate(cropped_img, skew_angle, cval=1)

                # resized_img = cv2.resize(rotated, (rotated.shape[1]//4, rotated.shape[0]//4))

                # rotated_2 = rotate(rotated, get_skew_angle(resized_img), cval=1)

                # img_with_grid = draw_grid(rotated.copy())

                cv2.imwrite(os.path.join(output_path, image), rotated*255)
                    
                print('-' * 50)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script with input and output paths as arguments")
    parser.add_argument("--input_path", type=str, nargs='?', help="Path to the directory containing input images", default='src/page_dewarp/example_input')
    parser.add_argument("--output_path", type=str, nargs='?', help="Path to the directory where output images will be saved", default='src/page_dewarp/example_output')
    parser.add_argument("--cleanup", action='store_true', help="Delete the contents of the unwarping_output folder")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.cleanup)
