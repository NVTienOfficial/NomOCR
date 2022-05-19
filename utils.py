import numpy as np
import argparse
import torch
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms
from PIL import Image
import numpy as np
from skimage.draw import rectangle_perimeter
from PIL import Image, ImageFont,ImageDraw


MODEL_PATH = 'models/best_pred_ocr2.h5'
from models.VGG import VGG
with open("labels/hex.txt", "r", encoding = 'utf-8') as f:
    labels = f.readlines()


input_size = 512
output_size = 128

def make_square(im, fill_color=(255, 255, 255, 0)):
    x, y = im.size
    size = max( x, y)
    new_im =  Image.new('L', (size, size), color=255)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def _nms( img, predict, nms_score, iou_threshold, reg):
    reg_model =  VGG((40,40,1),'categorical_crossentropy')
    reg_model.load_weights('models/checkpoint/VGG.h5')
    bbox = list()
    score_list = list()
    im_draw = np.asarray(torchvision.transforms.functional.resize(img, (img.size[1], img.size[0]))).copy()
    
    heatmap=predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]
    
    
    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:

        row = j // output_size 
        col = j - row*output_size
        
        bias_x = offset_x[row, col] * (img.size[1] / output_size)
        bias_y = offset_y[row, col] * (img.size[0] / output_size)

        width = width_map[row, col] * output_size * (img.size[1] / output_size)
        height = height_map[row, col] * output_size * (img.size[0] / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img.size[1] / output_size) + bias_y
        col = col * (img.size[0] / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])
        
    _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)), iou_threshold=iou_threshold)
    
    for k in range(len(_nms_index)):
    
        top, left, bottom, right = bbox[_nms_index[k]]
        
        ###########
        img_letter=img.crop((int(left),int(top),int(right),int(bottom)))#.resize((32,32)
        img_letter=img_letter.convert('L')
        img_letter=make_square(img_letter)
        img_letter= img_letter.resize((40,40))
        img_letter=img_letter.convert('L')
        predict=reg_model.predict(np.asarray(img_letter).reshape(1,40,40,1)/255)
        
        index=np.argmax(predict)
#         print(str(right-left))        
#         print(str(top-bottom))
        image = Image.new('L', (int(right-left), int(bottom-top)), color=255)
        

        font = ImageFont.truetype('static/fonts/NomNaTong-Regular.ttf', int(min( (right-left),(bottom-top) )*3/5) )
        drawing = ImageDraw.Draw(image)
        w, h = drawing.textsize( chr(int(labels[index],16)), font=font)
        drawing.text(
        ((right-left-w)/2, (bottom-top-h)/2),
                chr(int(labels[index],16)),
                fill=(0),
                font=font
            )
        
#         c_x=int((left+right)/2)
#         c_y=int((top+bottom)/2)
        c_x=int(left)
        c_y=int(top)
        offset=(c_x,c_y)
        if(predict[0,index]>=0.35):
            reg.paste(image, offset)
        ###########################
        start = (top, left)
        end = (bottom, right)
        
        rr, cc = rectangle_perimeter(start, end=end,shape=(img.size[1], img.size[0]))
        
        im_draw[rr, cc] = (255, 0, 0)
        
    return im_draw,reg