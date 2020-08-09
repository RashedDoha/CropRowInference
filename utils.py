import cv2
import torch
import numpy as np
from scipy.ndimage.measurements import label
from models.tiramisu.tiramisu import DenseUNet
from albumentations.pytorch import ToTensor
from albumentations import Normalize, CenterCrop, Compose

def get_video(video_path):
    try:
        video = cv2.VideoCapture(video_path)
    except:
        raise FileNotFoundError('Could not find video')
    return video

def get_model(model_arch, model_path, device='cuda'):
    if model_arch == 'densenet':
        model = DenseUNet(nb_classes=2)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

def crop_image(img):
    cropper = CenterCrop(128,256)
    return cropper(image=img)['image']

def get_transformed_img(img):
    transforms = Compose([CenterCrop(128,256),
                          Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
                          ToTensor()])
    return transforms(image=img)['image']

def get_prediction(model, image, device):
    if len(image.shape) < 4:
        image.unsqueeze_(0)
    image = image.to(device)
    return model(image)

def binarize(prediction):
    npimg = prediction.clone()
    if npimg.requires_grad:
        npimg = npimg.detach()
    if npimg.is_cuda:
        npimg = npimg.cpu()
    binary = npimg.type(torch.LongTensor).numpy()[0][1]
    vals = np.unique(binary)
    binary[binary < vals[-2]] = 0
    binary[binary != 0] = 1
    return binary

def get_rows(img, last_coefs=None, vis=True):
    pred_rows = []
    emp = np.zeros((128, 256)).astype(np.int32)
    structure = np.ones((3,3), dtype=np.int)
    labeled, ncomponents = label(img, structure)
#     assert ncomponents == 3, 'Number of components should be 3'
    labelens = []
    for i in range(1,labeled.max()+1):
        labelens.append((labeled == i).sum())
    center_rows = np.argsort(labelens)[-3:] + 1
    for k in center_rows:
        xs = []
        ys = []
        for i,j in zip(*np.where(labeled == k)):
            ys.append(j)
            xs.append(i)
        coefs = np.polyfit(xs, ys, deg=1)
        if last_coefs is None:
            diff1, diff2 = 0.0,0.0
        else:
            diff1, diff2 = abs(np.mean(coefs[0]-last_coefs[0])), abs(np.mean(coefs[1]-last_coefs[1]))
        
        diff = (diff1, diff2)

        alpha = 0.88
        if last_coefs is not None:
            coefs[:][0] = coefs[:][0]*alpha + last_coefs[:][0]*(1-alpha)
            # coefs[:][1] = last_coefs[:][1]
        row = [coefs[0]*x + coefs[1] for x in range(128)]
        row = [r for r in row if r <= 255]
        row = [r for r in row if r >= 0]
        row = np.floor(row).astype(np.int32)
        emp[np.arange(len(row)), row] = 1
        pred_rows.append(row)
    if vis:
        plt.imshow(emp, cmap='gray')
    return diff, coefs, labeled, pred_rows

def view_predicted_annotation(model,img,device, coefs=None):
    image = get_transformed_img(img)
    prediction = get_prediction(model, image, device)
    binary = binarize(prediction)
    diff, coefs, labeled, pred_rows = get_rows(binary, coefs, vis=False)
    def sortByPos(row):
        return row.mean()
    pred_rows.sort(key=sortByPos)
    t_img = crop_image(img)
    for i,row in enumerate(pred_rows):
        if i == 0:
            t_img[np.arange(len(row)), row, 0] = 0
            t_img[np.arange(len(row)), row, 1] = 0
            t_img[np.arange(len(row)), row, 2] = 255
        if i == 1:
            t_img[np.arange(len(row)), row, 0] = 211
            t_img[np.arange(len(row)), row, 1] = 0
            t_img[np.arange(len(row)), row, 2] = 148
        if i == 2:
            t_img[np.arange(len(row)), row, 0] = 255
            t_img[np.arange(len(row)), row, 1] = 0
            t_img[np.arange(len(row)), row, 2] = 0
    return diff, coefs, t_img

def write_video(frames, fps, video_dst):
    size = frames[0].shape
    size = size[1],size[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_dst, fourcc, float(fps), size)
    for i in range(len(frames)):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    
