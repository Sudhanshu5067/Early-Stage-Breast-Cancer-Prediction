from flask import Flask,request,render_template,flash,send_from_directory
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import PIL.Image

app=Flask(__name__)
app.config['SECRET_KEY']='Lakshmi'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/upload", methods=["POST","GET"])
def upload():
    if request.method=='POST':
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('images/', fn)
        myfile.save(mypath)
        print(fn)
        print(type(fn))
        accepted_formated=['jpg','png','jpeg','jfif','tif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted","Danger")
            return render_template('index.html')
        new_model = load_model("alg/FinalModel.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model. predict(test_image)
        classes=['No Cancer','Breast Cancer']
        prediction = classes[np.argmax(result)]


    return render_template("upload.html", image_name=fn, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)



if __name__=='__main__':
    app.run(debug=True)