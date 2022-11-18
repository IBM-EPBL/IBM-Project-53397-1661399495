import os.path
import pickle
import cv2
from flask import render_template, request
from skimage import feature
from flask import Flask

app = Flask(__name__)




@app.route("/Home")
def home():
    return render_template("Home.html")

@app.route("/Info")
def information():
    return render_template("Info.html")

@app.route("/upload")
def test():
    return render_template("Predict.html")

@app.route('/predict', methods=['GET','POST'])
def upload():

    if __name__ == "__main__":
        app.run(host='0.0.0.0',port=8000,debug=False)
    if request.method == 'POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)
        print("[INFO] loading model")
        model = pickle.loads(open('parkinson.pkl',"rb").read())
        image = cv2.imread(filepath)
        output = image.copy()
        output = cv2.resize(output,(128,128))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(200,200))
        image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = feature.hog(image,orientations =9,pixels_per_cell = (10,10) ,cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")
        preds = model.predict([features])
        print(preds)
        ls=["healthy","parkinson"]
        result = ls[preds[0]]

        color =(0,255,0) if result == "healthy" else (0,0,255)
        cv2.putText(output,result,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.imshow("Output",output)
        cv2.waitKey(0)
        return result
    return None
