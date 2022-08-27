import os
import cv2
import random
import string
import numpy as np

from flask import Flask
from flask import redirect
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename

from glob import glob
from tqdm import tqdm
from vars import Env
from datetime import datetime
from tensorflow.keras.models import load_model
from apscheduler.schedulers.background import BackgroundScheduler

random.seed()

HOME_ADDRESS = Env.HOME_ADDRESS
UPLOAD_FOLDER = Env.UPLOAD_FOLDER
MODEL_LOCATION = Env.MODEL_LOCATION
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1000 * 1000
app.config['SECRET_KEY'] = Env.SECRET_KEY


def cleanStorage():
    files = glob(os.path.join(UPLOAD_FOLDER, '*.*'))
    for file in tqdm(files):
        os.remove(file)
    print('=== STORAGE FOLDER IS CLEAN ===')


def checkFilesExtensions(image_name):
    return '.' in image_name and image_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generateSeed():
    current_time = datetime.now()
    time = current_time.strftime('%M:%S.%f')
    return time.split('.')[1]

def generateRandomString(image_name):
    random.seed(int(generateSeed()))
    image_ext = image_name.split('.')[-1]
    length = random.randint(5, 40)
    word = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    word = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_') + word + '.' + image_ext
    return word
    

@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        return render_template('/app/index.html')

    elif (request.method == 'POST'):
        if 'file' not in request.files:
            return redirect(HOME_ADDRESS)
        
        image = request.files['file']
        model = request.form['model']
        if image.filename == '':
            return redirect(HOME_ADDRESS)
        if image and checkFilesExtensions(image.filename):
            image_name = generateRandomString(secure_filename(image.filename))
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

            if (model == 'CNN'):
                model_classify = load_model(os.path.join(MODEL_LOCATION, 'model_cnn.h5'))
            elif (model == 'CNN_im'):
                model_classify = load_model(os.path.join(MODEL_LOCATION, 'model_cnn_improved.h5'))
            elif (model == 'Xception'):
                model_classify = load_model(os.path.join(MODEL_LOCATION, 'model_xception.h5'))
                
            image_path = glob(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
            image = cv2.imread(image_path[0])
            image = cv2.resize(image, (200, 200))
            image = np.expand_dims(image, axis=0)

            prediction = model_classify.predict(image)

            return render_template('app/test.html', result = int(prediction))

    return redirect(HOME_ADDRESS)

if __name__ == '__main__':
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(cleanStorage, 'interval', minutes=15)
    scheduler.start()

    app.run('localhost', 9000, debug=True)

    scheduler.shutdown()
