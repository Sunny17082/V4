from flask import Flask, request, render_template,  redirect, session
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd




app = Flask(__name__)


# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def dataset(user_input):
    d = pd.read_csv('data.csv')

    data = {}
    recom = {}
    data['image'] = f"../static/img/{list(d[d['name'] == user_input]['image'])[0]}"
    data['water'] = list(d[d['name'] == user_input]['water'])[0]
    data['blue'] = list(d[d['name'] == user_input]['blue'])[0]
    data['green'] = list(d[d['name'] == user_input]['green'])[0]
    data['grey'] = list(d[d['name'] == user_input]['grey'])[0]
    data['serving_size'] = list(d[d['name'] == user_input]['serving_size'])[0]
    data['name'] = user_input.title()
    if list(d[d['name'] == user_input]['recommendation'])[0] == 1:
        user_input = list(d[d['name'] == user_input]['recommendation_name'])[0]
        recom['image'] = f"../static/img/{list(d[d['name'] == user_input]['image'])[0]}"
        recom['water'] = list(d[d['name'] == user_input]['water'])[0]
        recom['blue'] = list(d[d['name'] == user_input]['blue'])[0]
        recom['green'] = list(d[d['name'] == user_input]['green'])[0]
        recom['grey'] = list(d[d['name'] == user_input]['grey'])[0]
        recom['serving_size'] = list(d[d['name'] == user_input]['serving_size'])[0]
        recom['name'] = user_input.title()
        
        return [data, recom]
    return [data]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    return render_template('dashboard.html')


@app.route('/search', methods=['POST'])
def search_page():
    return render_template('search.html')



@app.route('/search_object', methods=['POST'])
def process_form():
    if request.method == 'POST':
        user_input = request.form['objectname']
        user_input = user_input.lower().strip()
        print(user_input)
        result = dataset(user_input)
        print(result)
        if len(result) == 1:
            return render_template('object.html', result=result[0])
        return render_template('objectWithRecom.html', result=result[0], recom=result[1])
    return "Error"

@app.route('/upload', methods=['POST'])
def upload_page1():
    return render_template('upload.html')



@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)


        img=tf.keras.preprocessing.image.load_img(file_path,target_size=(224,224))
        model = tf.keras.models.load_model('Toothpaste_Toothbrush_Apple_Banana_Grape_V2.h5')

        x=tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add a batch dimension
        x = x / 255.0
        result = np.argmax(model.predict(x), axis=1)
        if result[0] == 0:
            user_in = 'apple'
        elif result[0] == 1:
            user_in = 'banana'
        elif result[0] == 2:
            user_in = 'grape'
        elif result[0] == 3:
            user_in = 'toothbrush'
        elif result[0] == 4:
            user_in = 'toothpaste'

        data = dataset(user_in)
        print(data)
        if len(data) == 1:
            return render_template('object.html', result=data[0])
        return render_template('objectWithRecom.html', result=data[0], recom=data[1])
    

if __name__ == '__main__':
    app.run(debug=True)
