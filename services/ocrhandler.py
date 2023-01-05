import os
import cv2
import numpy as np
import pytesseract
from pathlib import Path
import logging
from config.config import OUTPUT_DIR
from logging.config import dictConfig
import math
import requests
from PIL import ImageFont, ImageDraw, Image

log = logging.getLogger('file')

class BOXES_HELPER():
    def get_organized_tesseract_dictionary(self, tesseract_dictionary):
        res = {}
        log.info(f"Tesseract Dictionary {tesseract_dictionary}")
        n_boxes = len(tesseract_dictionary['level'])

        # Organize blocks
        res['blocks'] = {}
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 2:
                res['blocks'][tesseract_dictionary['block_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'paragraphs': {}
                }

        # Organize paragraphs
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 3:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][
                    tesseract_dictionary['par_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'lines': {}
                }

        # Organize lines
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 4:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][tesseract_dictionary['par_num'][
                    i]]['lines'][tesseract_dictionary['line_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'words': {}
                }

        # Organize words
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 5:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][
                    tesseract_dictionary['par_num'][
                        i]]['lines'][tesseract_dictionary['line_num'][i]]['words'][tesseract_dictionary['word_num'][i]] \
                    = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'text': tesseract_dictionary['text'][i],
                    'conf': float(tesseract_dictionary['conf'][i]),
                }

        return res

    def get_lines_with_words(self, organized_tesseract_dictionary):
        res = []
        for block in organized_tesseract_dictionary['blocks'].values():
            for paragraph in block['paragraphs'].values():
                for line in paragraph['lines'].values():
                    if 'words' in line and len(line['words']) > 0:
                        currentLineText = ''
                        for word in line['words'].values():
                            if word['conf'] > 60.0 and not word['text'].isspace():
                                currentLineText += word['text'] + ' '
                        if currentLineText != '':
                            res.append(
                                {'text': currentLineText, 'left': line['left'], 'top': line['top'], 'width': line[
                                    'width'], 'height': line[
                                    'height']})

        return res

    def midpoint(self,x1, y1, x2, y2):
        x_mid = int((x1 + x2)/2)
        y_mid = int((y1 + y2)/2)
        return (x_mid, y_mid)

    def get_translation(self,text_input):
        url = 'https://meity-dev.ulcacontrib.org/aai4b-nmt-inference/v0/translate'
        parameters = {
                        "input": [
                        {
                            "source": text_input
                        }
                        ],
                        "config": {
                            "modelId":103,
                            "language": {
                                "sourceLanguage": "en",
                                "targetLanguage": "hi"
                            }
                        }
                    }
        x = requests.post(url, json = parameters)
        return x.json()['output'][0]['target']

    def show_boxes_lines(self, d, frame):
        text_vertical_margin = 12
        organized_tesseract_dictionary = self.get_organized_tesseract_dictionary(d)
        lines_with_words = self.get_lines_with_words(organized_tesseract_dictionary)
        # print(lines_with_words)
        for line in lines_with_words:
            if line['text'] == '': 
                continue
            x = line['left']
            y = line['top']
            h = line['height']
            w = line['width']
            #Modification: Remove the current text and replace it with surrounding color
            #left is x coordinate (Top left)
            #top is y coordinate (Top left)
            #height = difference between x coordinates
            #width = difference between y coordinates
            #confidence = % of confidence that it's a text

            #Mods - for enlarging area of text
            # line['left'] -= 15
            # line['top'] -= 15
            # line['width'] += 15
            # line['height'] += 15

            #Top left
            x1 = line['left']
            y1 = line['top']

            #Top right
            x2 = line['left'] + line['width']
            y2 = line['top']

            #Bottom left
            x3 = line['left']
            y3 = line['top'] + line['height']

            #Bottom Right
            x4 = line['left'] + line['width']
            y4 = line['top'] + line['height']

            x_mid0, y_mid0 = self.midpoint(x1, y1, x3, y3)
            x_mid1, y_mi1 = self.midpoint(x2, y2, x4, y4)
            thickness = int(math.sqrt( (x3 - x1)**2 + (y3 - y1)**2 ))
            
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
            #masked = cv2.bitwise_and(frame, frame, mask=mask)
            frame = cv2.inpaint(frame, mask, 7, cv2.INPAINT_NS)

            #Summary: Add text and rectangle around already existing text within the frame.
            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            translated_text = self.get_translation(line['text'])
            log.info(f"Text to be printed {translated_text}")

            fontpath = "./Fonts/TiroDevanagariHindi-Regular.ttf" # <== 这里是宋体路径 
            font = ImageFont.truetype(fontpath, 32)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y),  translated_text, font = font, fill = (0, 255, 0, 0))
            frame = np.array(img_pil)

            #Summary: CV2 Put Text to store text within frame
            # frame = cv2.putText(frame,
            #                     text=translated_text,
            #                     #org=(x, y - text_vertical_margin),
            #                     org=(x, y),
            #                     fontFace=cv2.FONT_HERSHEY_DUPLEX,
            #                     fontScale=0.5,
            #                     color=(0, 255, 0),
            #                     thickness=1)
        return frame

    def show_boxes_words(self, d, frame):
        text_vertical_margin = 12
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if (int(float(d['conf'][i])) > 60) and not (d['text'][i].isspace()):  # Words
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text=d['text'][i], org=(x, y - text_vertical_margin),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=1,
                                    color=(0, 255, 0), thickness=2)
        return frame


class OCR_HANDLER:

    def __init__(self, video_filepath, cv2_helper, ocr_type="WORDS"):
        # The video_filepath's name with extension
        self.video_filepath = video_filepath
        self.cv2_helper = cv2_helper
        self.ocr_type = ocr_type
        self.boxes_helper = BOXES_HELPER()
        self.video_name = Path(self.video_filepath).stem
        self.frames_folder = OUTPUT_DIR + 'temp/' + self.video_name + '_frames'
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'MP4V' if this doesn't work on your OS.
        self.out_extension = '.mp4'
        self.out_name = self.video_name + '_boxes' + self.out_extension

    ########## EXTRACT FRAMES AND FIND WORDS #############
    def process_frames(self):

        frame_name = './' + self.frames_folder + '/' + self.video_name + '_frame_'

        if not os.path.exists(self.frames_folder):
            os.makedirs(self.frames_folder)

        video = cv2.VideoCapture(self.video_filepath)
        self.fps = round(video.get(cv2.CAP_PROP_FPS))  # get the FPS of the video_filepath
        frames_durations, frame_count = self.get_saving_frames_durations(video, self.fps)  # list of point to save

        log.info("SAVING VIDEO:", frame_count, "FRAMES AT", self.fps, "FPS")

        idx = 0
        while True:
            is_read, frame = video.read()
            if not is_read:  # break out of the loop if there are no frames to read
                break
            frame_duration = idx / self.fps
            try:
                # get the earliest duration to save
                closest_duration = frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, then save the frame
                output_name = frame_name + str(idx) + '.png'
                frame = self.ocr_frame(frame)
                cv2.imwrite(output_name, frame)

                if (idx % 10 == 0) and (idx > 0):
                    log.info(f"Saving frame: {output_name} with index {idx}, frame_duration {frame_duration} and closest_duration {closest_duration}")
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            idx += 1
        #if (idx - 1 % 10 != 0):
            #print(">")
        log.info("\nSaved and processed", idx, "frames")
        video.release()

    def assemble_video(self):

        print("ASSEMBLING NEW VIDEO")

        images = [img for img in os.listdir(self.frames_folder) if img.endswith(".png")]  # Careful with the order
        images = sorted(images, key=lambda x: float((x.split("_")[-1])[:-4]))

        frame = cv2.imread(os.path.join(self.frames_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(OUTPUT_DIR + self.out_name, self._fourcc, self.fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.frames_folder, image)))

        video.release()

        # When finished, delete all frames stored temporarily on disk.
        for f in os.listdir(self.frames_folder):
            if not f.endswith(".png"):
                continue
            try:
                os.remove(os.path.join(self.frames_folder, f))
            except OSError as e:
                print("Error: %s : %s" % (self.frames_folder, e.strerror))

        # Then delete the directory that contained the frames.
        try:
            os.rmdir(self.frames_folder)
        except OSError as e:
            print("Error: %s : %s" % (self.frames_folder, e.strerror))

    def get_saving_frames_durations(self, video, saving_fps):
        """A function that returns the list of durations where to save the frames"""
        s = []
        # get the clip duration by dividing number of frames by the number of frames per second
        clip_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        # use np.arange() to make floating-point steps
        log.info(f"Clip Duration in seconds: {clip_duration}")
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        #s is list of seconds where frame must be taken
        return s, video.get(cv2.CAP_PROP_FRAME_COUNT)

    def ocr_frame(self, frame):

        im, d = self.compute_best_preprocess(self.cv2_helper.get_grayscale(frame))
        if d is not None: 
            if (self.ocr_type == "LINES"):
                frame = self.boxes_helper.show_boxes_lines(d, frame)
            else:
                frame = self.boxes_helper.show_boxes_words(d, frame)
        return frame

    def compute_best_preprocess(self, frame):
        def f(count, mean):
            return 10 * count + mean

        best_f = 0
        best_opt = 0
        best_im = frame
        best_d = None
        options = [["binarization1"],["binarization2"],["binarization2","remove_noise","dilate"]]

        for idx, opt in enumerate(options):
            # Apply preprocess
            im = frame
            if "binarization1" in opt:
                im = self.cv2_helper.binarization_adaptative_threshold(im)
            if "binarization2" in opt:
                im = self.cv2_helper.binarization_otsu(im)
            if "remove_noise" in opt:
                im = self.cv2_helper.remove_noise(im)
            if "dilate" in opt:
                im = self.cv2_helper.dilate(im)

            # Compute mean conf:
            d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
            confs = [int(float(d['conf'][i])) for i in range(len(d['text'])) if not (d['text'][i].isspace())]
            confs = [i for i in confs if i > 60]

            mean_conf = np.asarray(confs).mean() if len(confs) > 0 else 0

            #print(len(confs),mean_conf,f(len(confs),mean_conf))

            if f(len(confs), mean_conf) > best_f:
                best_im = im
                best_d = d
                best_f = f(len(confs), mean_conf)
                #print(opt)

        return best_im, best_d

# Log config
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] {%(filename)s:%(lineno)d} %(threadName)s %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {
        'info': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'info.log'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        'file': {
            'level': 'DEBUG',
            'handlers': ['info', 'console'],
            'propagate': ''
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['info', 'console']
    }
})