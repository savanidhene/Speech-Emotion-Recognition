
from flask import Flask , render_template , session , request , redirect, url_for
import os
from werkzeug.utils import secure_filename
from delete_files import delete
from speech import recognize

UPLOAD_FOLDER = os.path.join("form-input")
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def main():
    return render_template('main.html')



@app.route('/search')
def search():
    return render_template('search.html')



@app.route('/result', methods = ['GET', 'POST'])
def upload_file():
    delete()
    if request.method == 'POST':
        # delete() 
        f = request.files['file']
        if f.filename == '':
                return render_template("result.html",name="No file selected!")
        #f.save(secure_filename(f.filename))
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ans = recognize()        
        
        return render_template("result.html",name="File uploaded successfully!", ans=ans)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

