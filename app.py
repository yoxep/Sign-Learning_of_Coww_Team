import errno
from flask import Flask, url_for, render_template, Response, request, redirect, g, session
from keras.utils.image_utils import img_to_array, load_img
from keras.models import load_model
import numpy as np
import json
import random
from flask_babel import Babel, gettext
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
import os

app = Flask(__name__)
babel = Babel()
babel.init_app(app)


app.config['lang_code'] = ['en', 'ko']

max_num_hands = 1
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

model = load_model('model/handtrain(500(88.2)).h5')

# crop_img 엉키지 않게
class Microsecond(object):
    def __init__(self):
        dt = datetime.now()
        self.microsecond = dt.microsecond

    
    def get_path_name(self):
        return 'model/' + str(self.microsecond)

crop_img_origin_path = Microsecond()
default_img = cv2.imread('model/crop_img.jpg')

# 새로운 폴더 만들기!
try:
    if not(os.path.isdir(crop_img_origin_path.get_path_name())):
        os.makedirs(os.path.join(crop_img_origin_path.get_path_name()))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

# make directory
origin_path = crop_img_origin_path.get_path_name() + '/crop_img.jpg'
cv2.imwrite(origin_path, default_img)
    
# h5 모델 
def get_label(idx):
    label = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ",
    "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"
]
    return label[idx]


# ** 전역변수 대신 클래스 객체 사용
class PredictLabel(object):
    def __init__(self, label):
        self.label = label

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

class Target_idx(object):
    def __init__(self, idx):
        self.idx = idx

    def set_idx(self, idx):
        self.idx = idx

    def get_idx(self):
        return self.idx

target_idx = Target_idx(0)
predict_label = PredictLabel('')


@app.before_request
def before_request():
    g.total_q = 10

def get_k_list():
    k_list = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ",
    "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"
]

    return k_list


def k_list_idx(element):
    next_topic = ""
    previous_topic = ""

    k_list = get_k_list()

    list_idx_end = len(k_list) - 1  # 마지막 인덱스
    idx_now = k_list.index(element)

    if idx_now == list_idx_end:
        next_topic = k_list[0]
    else:
        next_topic = k_list[idx_now + 1]

    if idx_now != 0:
        previous_topic = k_list[idx_now - 1]

    return next_topic, previous_topic


def gen(camera):
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            continue
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 30
        out = cv2.VideoWriter('video.avi', fourcc, fps, (int(width), int(height)))

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],:]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                result = model.predict([data]).squeeze()
                idx = np.argmax(result)

                img = Image.fromarray(img)

                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("fonts/gulim.ttc", 100)
                org = (300, 50)
                text = get_label(idx)
                draw.text(org, text, font=font, fill=(0, 0, 0))
                img = np.array(img)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        out.write(img)
        if cv2.waitKey(1) == ord('q'):
            break


def make_quiz():
    question_list = {}
    img_list = []
    for i in range(g.total_q):
        question, examples, img = make_random_quiz(question_list)
        question_list[question] = examples
        img_list.append(img)
    return question_list, img_list


def make_random_quiz(question_list):
    alphabet_list = get_k_list()
    examples = []  # 보기
    while True:
        answer = alphabet_list[random.randint(0, len(alphabet_list) - 1)]
        if is_valid_quiz(answer, question_list):
            break
    examples.append(answer)
    while len(examples) != 4:
        randomIndex = random.randint(0, len(alphabet_list) - 1)
        if alphabet_list[randomIndex] not in examples:
            examples.append(alphabet_list[randomIndex])
    random.shuffle(examples)
    img = []
    for i in examples:
        img.append('../static/img/asl_' + i + ".png")

    return answer, examples, img


# 이전에 낸 문제인지 확인
def is_valid_quiz(answer, question_list):
    if answer in question_list:
        return False
    else:
        return True


# for ajax
@app.route('/return_label', methods=['POST', 'GET'])
def return_label():
    value = request.form.get("target", False)
    # 띄어쓰기 조심!@!
    label_list = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ",
    "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"
]
    tem = label_list.index(value)
    print(tem)
    idx = target_idx.set_idx(tem)
    print(idx)
    label = " " + predict_label.get_label() + " "

    # ajax 에서 값 받아올때 공백이 앞뒤로 붙는데 python strip() 함수가 안먹어서...

    if label == '':
        predict_result = {
            'status': 0,
            'info': 'not detected',
            'label': '',
            'lang_code': session['language']
        
        }
    elif label != value:
        predict_result = {
            'status': 0,
            'info': gettext('predict_incorrect'),
            'label': label,
            'lang_code': session['language']

        }
        print("틀림!")
    else:
        predict_result = {
            'status': 1,
            'info': gettext('predict_correct'),
            'label': label,
            'lang_code': session['language']
        }

    # result 의 status 값이 1이면 참 -> main.js 에서 correct 값 증가

    json_data = json.dumps(predict_result)  # json 형태로 바꿔줘야 에러 안남
    return json_data


# for ajax
@app.route('/english')
def english():
    session['language'] = 'en'
    link = request.args.get('link')
    if link:
        return redirect(link)
    else:
        return redirect('/')


# for ajax
@app.route('/korean')
def korean():
    session['language'] = 'ko'
    link = request.args.get('link')
    if link:
        return redirect(link)
    else:
        return redirect('/')


@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if request.method == 'GET':
        question_list, img_list = make_quiz()
        return render_template('quiz.html', str=str, enumerate=enumerate, question_list=question_list,
                               img_list=img_list, total_q=g.total_q, link=request.full_path)

    if request.method == 'POST':
        user_answers = {}
        for i in range(g.total_q):
            question = "question" + str(i)
            answer = "answer" + str(i)
            q = request.form[question]
            a = request.form[answer]
            user_answers[q] = a
            print(user_answers)
        user_answers = json.dumps(user_answers)

        return redirect(url_for('.quiz_result', user_answers=user_answers))


# 퀴즈 결과
@app.route('/quiz/result')
def quiz_result():
    try:
        user_answers = json.loads(request.args['user_answers'])
        items = user_answers.items()
    except:
        user_answers = {}
    items = user_answers.items()
    correct_num = 0
    incorrect_questions = []
    for q, a in items:
        if (q == a):
            correct_num += 1
        else:
            incorrect_questions.append(q.upper())
    if correct_num == g.total_q:
        img_path = "../static/img/score_100.png"
    elif correct_num >= (g.total_q // 2):
        img_path = "../static/img/score_50.png"
    else:
        img_path = "../static/img/score_0.png"
    return render_template('result.html', correct_num=correct_num, incorrect_questions=incorrect_questions,
                           total_q=g.total_q, img_path=img_path, link=request.full_path)


@app.route('/practice_asl')
def practice_asl():
    alphabet_list = get_k_list()
    return render_template('practice_asl.html', alphabet_list=alphabet_list, link=request.full_path)


# video streaming
@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/practice', methods=['GET', 'POST'])
def practice():
    element = request.args.get('element')
    alphabet = element.upper()
    img = "../static/img/asl_" + element + ".png"

    next_topic, previous_topic = k_list_idx(element)

    return render_template('practice.html', img=img, alphabet=alphabet, previous_topic=previous_topic,
                           next_topic=next_topic, link=request.full_path)


@app.route('/')
def index():
    return render_template('main.html', link=request.full_path)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5000, debug=True)