from flask import Flask,render_template,redirect,request,send_from_directory
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
model_file = "model.h5"
model = load_model(model_file)

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def makePredictions(path):
    img = Image.open(path)
    img_d = img.resize((224,224))
    img = np.array(img_d,dtype=np.float64)
    img = img.reshape((1,224,224,3))
    predictions = model.predict(img)
    a = int(np.argmax(predictions))
    if a==1:
        a = "pneumonic"
    else:
        a="healthy"
    return a

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        if 'img' not in request.files:
            return render_template('home.html',filename="unnamed.png",message="Please upload an file")
        f = request.files['img']  
        if f.filename=='':
            return render_template('home.html',filename="unnamed.png",message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html',filename="unnamed.png",message="please upload an image with .png or .jpg/.jpeg extension")
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        return render_template('home.html',filename=f.filename,message=predictions,show=True)
    return render_template('home.html',filename='unnamed.png')

if __name__=="__main__":
    app.run(debug=True)