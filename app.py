from http.client import OK
import numpy as np
import cv2
import os
#from utils import *
from utils import _nms
import json
from flask import Flask, request, render_template, redirect, send_file
from tensorflow.keras.models import load_model
import argparse
import torch
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms 
from PIL import Image 
import numpy as np
from skimage.draw import rectangle_perimeter
#from flask_caching import Cache

from models.HRCenterNet import HRCenterNet
from tool.denoising_and_bens_preprocessing import denoise_and_bens_prepro

input_size = 512
output_size = 128

test_tx = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('divece: ', device)
checkpoint = torch.load('models/checkpoint/HRCenterNet.pth.tar', map_location="cpu")    
    
model = HRCenterNet()
model.load_state_dict(checkpoint['model'])
model.eval()

if torch.cuda.is_available():
    model.cuda()
# Initialize
if os.path.exists("static/save/last.jpg"):
    os.remove("static/save/last.jpg")
else:
    print("The file does not exist")
#cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
app = Flask(__name__)
#cache.init_app(app)

# # clear cache
# with app.app_context():
#     cache.clear()
    
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 512 #16KB
index=0

@app.route('/')
def home():
    
    return render_template('index.html')
@app.route('/', methods=['POST'])
#@cache.memoize(timeout=100)
def predict():
    global index
    #cache.delete_memoized(predict)
    if 'file' not in request.files:
        print('No file')
        if os.path.exists(f"static/save/last_{index}.jpg"):
            os.remove(f"static/save/last_{index}.jpg")
        else:
            print("The file does not exist") 
        return redirect(request.url)
#     else:
#         with app.app_context():
#         cache.clear()
    file = request.files['file']
    if file.filename == '':
        print('Empty file!')
        if os.path.exists(f"static/save/last_{index}.jpg"):
            os.remove(f"static/save/last_{index}.jpg")
        else:
            print("The file does not exist")
        return redirect(request.url)
    print('load successfully')
    if file:
        print(file)
        img_str = file.read()
        nparr = np.frombuffer(img_str, np.uint8)
        #print('aaaaaaaaaa')
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR )
        for file in os.listdir('static/save'):
            if file.endswith('.jpg'):
                os.remove('static/save/'+file) 
        cv2.imwrite(f'static/save/last_{index}.jpg', image)
        image = Image.open(f'static/save/last_{index}.jpg').convert("RGB")
        denoised = denoise_and_bens_prepro(image)
        image_tensor = test_tx(denoised)
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        inp = inp.to(device, dtype=torch.float) 
        predict = model(inp) 
        reg=  Image.new('RGB', image.size)
        out_img,reg = _nms( image, predict, nms_score=0.3, iou_threshold=0.1, reg=reg)
        Image.fromarray(out_img).save(f'static/save/dec_{index}.jpg')
        Image.fromarray(np.asarray(reg)).save(f'static/save/reg_{index}.jpg')
        #test=Image.fromarray(out_img).show()
        my_dict = {
        'dec': f'static/save/dec_{index}.jpg',
        'reg': f'static/save/reg_{index}.jpg'
        
        }
        x = json.dumps(my_dict)
        index+=1
        
        return  x
        # return render_template('index.html', user_image = 'static/save/test.jpg')

@app.route("/image",methods=["GET"])
def getImage():
    # try:
    #     image = cv2.imread('static/save/page01a.jpg')
    #     print(image)    
    #     return send_file(image, mimetype='image/gif')
    # except Exception as e:
    #     print(e)
    # return "O"
    return "static/save/test.jpg"
    
if __name__ == "__main__":
    app.run(debug=True)
    for file in os.listdir('static/save'):
        if file.endswith('.jpg'):
             os.remove('static/save/'+file) 