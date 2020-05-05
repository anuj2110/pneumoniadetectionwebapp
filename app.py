from flask import Flask,render_template,redirect,request,send_from_directory
import os
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
@app.route('/success',methods = ['POST'])
def success():
    if request.method=='POST':
        f = request.files['img']  
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))  
        return render_template('success.html',filename=f.filename)

if __name__=="__main__":
    app.run(debug=True)