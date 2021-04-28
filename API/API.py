import os

import redirect as redirect
import self as self

from preprocessor_andAnalyser_lib import preprocessor_andAnalyser_lib
from flask import Flask, request, jsonify # Import the flask web server
from werkzeug.utils import secure_filename

app = Flask(__name__) # Single module that grabs all modules executing from this file

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file :
            ppa = preprocessor_andAnalyser_lib();

            filename = secure_filename(file.filename)
            file.save(os.path.join(ppa.root_folder_for_uploads, filename))

            frame, target_data_ = ppa.prepare_file_to_frame(os.path.join(ppa.root_folder_for_uploads, filename))
            images = ppa.convert_images(frame)
            json = ppa.classify(frame,images,target_data_)
            json_dictionary = ppa.semantic_linking(json)

            return json_dictionary



# set FLASK_APP=API\API.py
# flask run
