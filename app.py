import os
import time
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from models.elm import run_from_model

# Configuration
UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # Define allowed image types

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Optional: Limit upload size (e.g., 16MB)
app.secret_key = 'my_secret_code'

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            flash('No file part selected.', 'error')
            return redirect(request.url)

        file = request.files['imagefile']

        if file.filename == '':
            flash('No selected file.', 'warning')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            try:
                file.save(filepath)
                flash(f'Image "{filename}" uploaded successfully!', 'success')

                classification_results = run_from_model(filepath)

                return render_template('index.html', result_string=classification_results, uploaded_filename=filename)

            except Exception as e:
                 flash(f'An error occurred during processing: {e}', 'error')
                 print(f"Error: {e}")
                 return redirect(url_for('home'))

        elif file and not allowed_file(file.filename):
            flash('Invalid file type. Allowed types: png, jpg, jpeg, gif', 'error')
            return redirect(request.url)

        else:
             flash('An unknown error occurred during upload.', 'error')
             return redirect(request.url)

    return render_template("index.html", results=None, uploaded_filename=None)


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created directory: {UPLOAD_FOLDER}")
    app.run(debug=True)