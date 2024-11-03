import os
from pathlib import Path
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

from segment import segment, save
from flaskdemo import reconstruct

# flask app set up
app = Flask(__name__)

# store images uploaded
app.config['UPLOAD_FOLDER'] = 'Deshredder/static/uploads'
app.config['FRAGMENTS_FOLDER'] = 'Deshredder/static/fragments/document/strips'
app.config['OUTPUT_FOLDER'] = 'Deshredder/static/output'

# allowed extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# check if a file name is well formated (png or jpg)
def allowed_file(filename):
    if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        return True
    return False

def generate_output(output_path='./Deshredder/static/output', width=800, height=600):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [str(f) for f in Path('./Deshredder/static/uploads').iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if len(image_files) >= 2:
        # print(image_files[0], image_files[1])
        front = cv2.imread(image_files[1])[..., ::-1]
        back = cv2.imread(image_files[0])[..., ::-1]
        strips, masks = segment(front, back, 50, 0.05, 3, 4, 0.5, 100, 200, 0, nCPU=12)
        save(strips, masks, docname='document', path='./Deshredder/static/fragments')
        print("Document segmented")
        reconstruct('./Deshredder/static/fragments/document', output_path)
        print(f"reconstruction saved as {output_path}/reconstruction.png")

    # # create a new default img
    # black_image = Image.new('RGB', (width, height), color='black')
    
    # # save it
    # black_image.save(output_path)
    # print(f"Black image saved as {output_path}")

    # # create a new default img
    # black_image = Image.new('RGB', (width, height), color='black')
    
    # # save it
    # black_image.save(output_path)
    # black_image.save('./static/fragments/hi.jpg')
    # black_image.save('./static/fragments/hi2.jpg')
    # print(f"Black image saved as {output_path}")

# main home page, GET is to see the web page, POST is trigger the deshredding process
@app.route("/", methods=["GET", "POST"])
def deshred():

    # initally there is no output to display
    output_image = None
    current_fragments = None

    if request.method == "POST":

        # option 1: the add fragment button was pressed
        if 'add_image' in request.form:

            # check that a file was uploaded first
            if 'image' not in request.files:
                return render_template("index.html", error="No file uploaded", images=[])
            
            # get the image uploaded and save it
            file = request.files['image']
            if file and allowed_file(file.filename):

                # create a secure version of the file name
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
        
        if 'deshred' in request.form:
            # TODO add deshredding code
            generate_output()
            output_image = 'reconstruction.png'
            current_fragments = os.listdir(app.config['FRAGMENTS_FOLDER'])
        
        if 'clear' in request.form:
            # clear uploads
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

            # clear fragments
            for file in os.listdir(app.config['FRAGMENTS_FOLDER']):
                os.remove(os.path.join(app.config['FRAGMENTS_FOLDER'], file))

            # clear outputs
            for file in os.listdir(app.config['OUTPUT_FOLDER']):
                os.remove(os.path.join(app.config['OUTPUT_FOLDER'], file))

    # render current images
    current_images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template("index.html", images=current_images, fragments=current_fragments, output_image=output_image)

if __name__ == "__main__":
    app.run(debug=True)

