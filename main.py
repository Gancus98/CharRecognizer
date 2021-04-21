#
#   Garnczarek Dawid - Programowanie Zaawansowane 2 - 02.2021
#
#   The program uses a webcam and a previously trained model to
#   recognizes letters, numbers and mathematical signs. If is an "=" sign in the recognized frame, program
#   tries to perform a math operation (only two-argument operations are allowed)
#
#   In addition, the program has the function of creating your own training set in the application,
#   select the appropriate char and then place it in the frame and save in trainData.csv.
#
########################################################################################

import string
import cv2
from tkinter import *
import PIL
from PIL import ImageTk
import tensorflow as tf
import numpy as np
import imutils
import csv


MODEL_NAME = 'model_final.h5'
START_BUTTON_IMG_PATH = 'img/start_button.png'
STOP_BUTTON_IMG_PATH = 'img/stop_button.png'
SAVE_BUTTON_IMG_PATH = 'img/save_csv_button.png'
DICTIONARY_BUTTON_IMG_PATH = 'img/dict_button_2.png'

IMG_SHAPE = (1, 28, 28)

nr_to_letter = {k: v.upper() for k, v in enumerate(list(string.ascii_lowercase))}
numbers_dict = {
    26: '1',
    27: '2',
    28: '3',
    29: '4',
    30: '5',
    31: '6',
    32: '7',
    33: '8',
    34: '9',
    35: '0',
    36: '=',
    37: '+',
    38: '-',
    39: '*',
    40: '/',
}
nr_to_letter.update(numbers_dict)
model = tf.keras.models.load_model(MODEL_NAME)


def predict_char_from_img(image):
    tensor = np.reshape(image, IMG_SHAPE)
    predict = model.predict(tensor)
    val = np.argmax(predict)
    predicted_character = nr_to_letter.get(val)
    return predicted_character


class CharacterDetector(object):
    PRIMARY_COLOR = '#122029'
    SECONDARY_COLOR = '#71c49a'
    CAMERA_WIDTH = 800
    CAMERA_HEIGHT = 600

    def __init__(self):
        self.is_predict_mode = 0
        self.init_root()
        self.load_images()
        self.init_app_layout()
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.setup_camera()
        self.main_loop()

    def main_loop(self):
        self.chars_img_to_save = []
        gray_image, preview_image = self.get_img_from_camera()

        if self.is_predict_mode:
            img_edges = self.prepare_frame_to_contours_detection(gray_image)
            detected_contours = self.detect_contours(img_edges)
            chars_img_ready_to_prediction, chars_boxes = self.prepare_chars_img_to_prediction(detected_contours,
                                                                                              gray_image)
            predicted_chars = self.predict_chars(chars_img_ready_to_prediction)
            boxes = self.get_boxes(chars_boxes)
            self.put_boxes_on_preview(predicted_chars, boxes, preview_image)
            self.convert_to_recognized_text_area(boxes, predicted_chars)

        self.put_frame_on_window(preview_image)

    def toggle_mode(self):
        if self.toggle_button.config('relief')[-1] == 'sunken':
            self.toggle_button.config(relief="raised", image=self.start_button_img)
            self.is_predict_mode = 0
        else:
            self.toggle_button.config(relief="sunken", image=self.stop_button_img)
            self.is_predict_mode = 1

    def setup_camera(self):
        self.cap = cv2.VideoCapture(1)  # chose camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)

    def get_img_from_camera(self):
        _, frame = self.cap.read()
        preview_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2GRAY)
        preview_image = cv2.rotate(preview_image, cv2.ROTATE_90_CLOCKWISE)
        gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
        return gray_image, preview_image

    def prepare_frame_to_contours_detection(self, gray_image):
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)  # for noise reduction
        return cv2.Canny(blurred, 30, 120, 4)

    def save_train_data(self):
        char_nr = self.get_char_key()
        with open(r'trainData.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for x in self.chars_img_to_save:
                single_row = [int(char_nr)]
                single_row.extend(np.reshape(x, [784]))
                writer.writerow(single_row)

    def get_char_key(self):
        char = self.variable.get()
        for x in nr_to_letter.items():
            if x[1] == char:  # 0-key 1-value
                return x[0]

    def put_frame_on_window(self, preview_image):
        preview_pil = PIL.Image.fromarray(preview_image)
        preview_tk = ImageTk.PhotoImage(image=preview_pil)
        self.cameraView.preview_tk = preview_tk
        self.cameraView.configure(image=preview_tk)
        self.cameraView.after(33, self.main_loop)

    def detect_contours(self, img_edges):
        detected_contours = cv2.findContours(img_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # only outer contours / only end points
        detected_contours = imutils.grab_contours(detected_contours)
        try:
            detected_contours.sort(key=lambda x: self.get_contour_precedence(x, 28))
            return detected_contours
        except:
            print("error")

    def get_contour_precedence(self, contour, cols):
        tolerance_factor = 60
        origin = cv2.boundingRect(contour)  # (x,y,w,h)
        return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

    def get_boxes(self, chars_boxes):
        return [b[1] for b in chars_boxes] # char_boxes[i][1] = x,y,w,h

    def prepare_chars_img_to_prediction(self, detected_contours, gray_image):
        chars_img_ready_to_prediction = []
        chars_boxes = []
        try:
            for c in detected_contours:
                (x, y, w, h) = cv2.boundingRect(c)
                if (15 <= w <= 60) or (15 <= h <= 60):
                    char_extracted = gray_image[y: y+h, x:  x+w]
                    char_extracted_after_threshold = cv2.threshold(char_extracted, 0, 255,
                                                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # threshold return 2 args 0-threshold 1-img_after_threshold
                    (char_height, char_width) = char_extracted_after_threshold.shape
                    if char_width > char_height:
                        char_extracted_after_threshold = imutils.resize(char_extracted_after_threshold, width=28)
                    else:
                        char_extracted_after_threshold = imutils.resize(char_extracted_after_threshold, height=28)

                    dX = int(max(0, 28 - char_width) / 2.0)
                    dY = int(max(0, 28 - char_height) / 2.0)

                    char_img_ready_to_prediction = cv2.copyMakeBorder(char_extracted_after_threshold, top=dY, bottom=dY,
                                                                      left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                                                      value=(0, 0, 0))
                    char_img_ready_to_prediction = cv2.resize(char_img_ready_to_prediction, (28, 28))
                    self.chars_img_to_save.append(char_img_ready_to_prediction)

                    char_img_ready_to_prediction = char_img_ready_to_prediction.astype("float32") / 255.0
                    char_img_ready_to_prediction = np.expand_dims(char_img_ready_to_prediction, axis=-1)
                    chars_boxes.append((char_img_ready_to_prediction, (x, y, w, h)))
                    chars_img_ready_to_prediction.append(np.reshape(char_img_ready_to_prediction, [28, 28]))
        except:
            print("preprare_chars_img_to_prediction - error")
        return chars_img_ready_to_prediction, chars_boxes

    def predict_chars(self, chars_img_ready_to_prediction):
        predicted_chars = []
        for x in chars_img_ready_to_prediction:
            predicted_chars.append(predict_char_from_img(x))
        return predicted_chars

    def put_boxes_on_preview(self, predicted_chars, boxes, preview_image):
        index = 0
        for (predicted_char, (x, y, w, h)) in zip(predicted_chars, boxes):
            cv2.rectangle(preview_image, (x, y), (x + w, y + h), (113, 196, 154), 2)
            cv2.putText(preview_image, predicted_chars[index], (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (113, 196, 154), 2)
            index += 1

    def convert_to_recognized_text_area(self, boxes, predicted_chars):
        is_math = 0
        for x in predicted_chars:
            if x == '=':
                is_math = 1
                break
        words = self.preprocess_for_text(boxes, predicted_chars, is_math)
        if is_math:
            try:
                string_solution = ""
                a = int(self.list_chars_to_string(words[0]))
                b = int(self.list_chars_to_string(words[2]))
                c = self.list_chars_to_string(words[1])

                if c == "+":
                    solution = a + b
                    string_solution = str(a) + '+' + str(b) + '=' + str(solution)
                if c == "-":
                    solution = a - b
                    string_solution = str(a) + '-' + str(b) + '=' + str(solution)
                if c == "*":
                    solution = a * b
                    string_solution = str(a) + '*' + str(b) + '=' + str(solution)
                if c == "/":
                    solution = a / b
                    string_solution = str(a) + '/' + str(b) + '=' + str(solution)
                self.textEntry.set(string_solution)
            except:
                self.textEntry.set("Error")
        else:
            self.textEntry.set(self.list_words_to_string(words))

    def preprocess_for_text(self,boxes, predicted_chars, is_math):
        single_word = []
        words = []
        last_char_x_w_position = 0
        last_char_y_h_position = 0
        position_x, position_y, char_width, char_height = 0, 1, 2, 3
        index = 0
        for x in boxes:
            if index == 0:
                last_char_x_w_position = x[position_x] + x[char_width]
                last_char_y_h_position = x[position_y] + x[char_height]
                single_word.append(predicted_chars[index])
            else:
                new_char_x_w_position = x[position_x] + x[char_width]
                new_char_y_h_position = x[position_y] + x[char_height]
                distance_x = new_char_x_w_position - last_char_x_w_position
                distance_y = new_char_y_h_position - last_char_y_h_position

                last_char_x_w_position = new_char_x_w_position
                last_char_y_h_position = new_char_y_h_position
                if distance_x < 45 and distance_y < 45 and index < len(boxes) - 1:
                    single_word.append(predicted_chars[index])
                else:
                    if index == len(boxes) - 1 and len(single_word) > 0:
                        if is_math:
                            words.append(single_word.copy())
                            single_word.clear()
                            single_word.append(predicted_chars[index])
                            words.append(single_word.copy())
                        else:
                            single_word.append(predicted_chars[index])
                            words.append(single_word.copy())
                    else:
                        words.append(single_word.copy())
                        single_word.clear()
                        single_word.append(predicted_chars[index])
            index += 1
        return words

    def list_chars_to_string(self, list_of_chars):
        connected_string = ""
        for x in list_of_chars:
            connected_string += x
        return connected_string

    def list_words_to_string(self, list_of_words):
        connected_string = ""
        for x in list_of_words:
            for y in x:
                connected_string += y
            connected_string += " "
        return connected_string

    def init_root(self):
        self.root = Tk()
        self.root.title('CharacterRecognizer')
        self.root.configure(borderwidth=25)
        self.root.geometry("540x790")
        self.root.resizable(width=False, height=False)

    def load_images(self):
        self.start_button_img = PhotoImage(file=START_BUTTON_IMG_PATH)
        self.stop_button_img = PhotoImage(file=STOP_BUTTON_IMG_PATH)
        self.save_button_img = PhotoImage(file=SAVE_BUTTON_IMG_PATH)
        self.dict_button_img = PhotoImage(file=DICTIONARY_BUTTON_IMG_PATH)

    def init_app_layout(self):
        self.background_label = Label(self.root, bg=self.PRIMARY_COLOR)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1, bordermode=OUTSIDE)

        self.cameraView = Label(self.root, highlightthickness=1)
        self.cameraView.grid(row=2, columnspan=6, pady=10)

        OPTIONS = list(nr_to_letter.values())
        OPTIONS.extend(["1","2","3","4","5","6","7","8","9","0","=","+","-","*","/"])
        self.variable = StringVar(self.root)
        self.variable.set(OPTIONS[0])  # default value
        self.option_menu = OptionMenu(self.root, self.variable, *OPTIONS)
        self.option_menu.config(borderwidth=0, image=self.dict_button_img, bg=self.PRIMARY_COLOR, highlightthickness=0,
                                indicatoron=0)
        self.option_menu.grid(row=0, column=4)

        self.save_train_data_bt = Button(self.root, command=self.save_train_data, borderwidth=0,
                                         image=self.save_button_img, bg=self.PRIMARY_COLOR)
        self.save_train_data_bt.grid(row=0, column=5, pady=5)

        self.textEntry = StringVar()
        self.text_area = Entry(self.root, textvariable=self.textEntry, width=60, bg=self.PRIMARY_COLOR, fg="white",
                               justify=CENTER, borderwidth=6)
        self.text_area.grid(row=1, column=0, columnspan=5, padx=10)

        self.toggle_button = Button(self.root, command=self.toggle_mode, borderwidth=0, image=self.start_button_img,
                                    bg=self.PRIMARY_COLOR)
        self.toggle_button.grid(row=1, column=5)


if __name__ == '__main__':
    CharacterDetector()
