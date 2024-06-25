# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from collections import Counter
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ['Forest', 'Glacier', 'Mountain','Sea','Street']



from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------
import cv2
def most_repeated_elements(lst):
    counter = Counter(lst)
    max_count = max(counter.values())
    most_repeated = [item for item, count in counter.items() if count == max_count]
    return most_repeated
app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Scene Recognition'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Scene Recognition'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            img = Image.open(file)
            img.save('output.BMP')


            prediction =pred_leaf_disease("output.BMP")

            prediction = (str(disease_dic[prediction]))

            print("The background scene is ",prediction)



            return render_template('scene-result.html', prediction="The Scenario is : ",precaution=prediction,title=title)
        #except:
         #   pass
    return render_template('scene.html', title=title)


@app.route('/video_upload')
def video_upload():
    title = 'Scene Recognition'
    return render_template('video.html', title=title)


@app.route('/upload', methods=['POST'])
def upload_file():
    title = 'Scene Recognition'
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'No selected file'
        
        if file:
            # Save the file to the static/images directory
            file.save('static/images/output.mp4')
            #return ('File uploaded successfully')


        video_path="static/images/output.mp4"
        cap = cv2.VideoCapture(video_path)

        # Initialize an empty list to store predictions
        frame_count = 0

        # List to store predictions
        predictions = []

        while cap.isOpened() and frame_count < 20:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL Image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_frame.save('output.BMP')

            # Predict disease on the frame
            prediction = pred_leaf_disease("output.BMP")
            
            prediction = (str(disease_dic[prediction]))

            print("The background scene is ", prediction)
            predictions.append(prediction)
            
            # Increment frame count
            frame_count += 1

        # Release the VideoCapture object and close any open windows
        print("all predictions are",predictions)
        #cap.release()
        #cv2.destroyAllWindows()
        most_repeated = most_repeated_elements(predictions)
        print("Most repeated elements:", most_repeated)        
        #return ('File uploaded successfully')
        return render_template('scene-result.html', prediction="The Scenario is : ",precaution=most_repeated,title=title)



# render disease prediction result page

@app.route('/real_time_classifcation', methods=['GET', 'POST'])
def real_time_classifcation():
    from real_time_predict  import capture_and_predict
    title = 'Scene Recognition'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            #file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            #img = Image.open(file)
            #img.save('output.BMP') 


            capture_and_predict()

           # prediction = (str(disease_dic[prediction]))

            #print("The background scene is ",prediction)



            return render_template('real_time_result.html', prediction="Real time Detection is  ",precaution="Success",title=title)
        #except:
         #   pass
    return render_template('real_time.html', title=title)






# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
