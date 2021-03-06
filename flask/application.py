import os
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from flask_face_recog import identify

UPLOAD_FOLDER = '/photo_uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super-secret'


@app.route('/')
def home():
    return render_template('home_page.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # make the path for the photo
            path = os.path.join('.', 'photo_uploads', filename)
            file.save(path)
            flash(identify(path))

            # return redirect(url_for('thank_you'))
    return render_template('face_recog.html')


@app.route('/thank_you/')
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
