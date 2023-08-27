
import torch
import torch.nn as nn
from torchvision import transforms,models
import argparse
import cv2
import json

def GetModel(modelName):
    if modelName == 'resnet18':
        model = models.resnet18(pretrained=False)
        nInputs = model.fc.in_features
        model.fc = nn.Sequential(
                nn.Linear(nInputs, 256), nn.ReLU(),
                nn.Linear(256, 3))
    if modelName == 'vgg16':
        model = models.vgg16(pretrained=False)
        nInputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
                nn.Linear(nInputs, 256), nn.ReLU(),
                nn.Linear(256, 3), nn.LogSoftmax(dim=1))    
    return model

def shotPredict(videoPath, modelPath,jsonPath, hoopCenter,hoopSize):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=128),
        #transforms.CenterCrop(size=112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model = GetModel('resnet18')
    model.load_state_dict(torch.load(modelPath))
    mps_device = torch.device("mps")

    model.to(mps_device)
    model.eval()

    cap = cv2.VideoCapture(videoPath)
    fps =  cap.get(cv2.CAP_PROP_FPS)
    if (cap.isOpened()== False):
        print("Error opening video file")



    count = 0
    signal = [0,0,0,0,0,0]
    flagNothing = True
    flagShot = False
    flagMade = False
    attemps = 0
    miss = 0
    made = 0
    display = False
    while(cap.isOpened()):
    
    # Capture frame-by-frame
        ret, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        dim = (int(width/2), int(height/2))
        if ret == True:
            frame = cv2.resize(frame,dim)
            cropImage = frame[(hoopCenter[1] - 10*hoopSize[1]): (hoopCenter[1] + 10*hoopSize[1]),(hoopCenter[0] - 3* hoopSize[0]):(hoopCenter[0] + 3* hoopSize[0])]
            cropImage_tensor = transform(cropImage)
            cropImage_tensor = cropImage_tensor.unsqueeze_(0)
            with torch.no_grad():
                data = cropImage_tensor.to(mps_device)
                out = model(data)

                if out[0][1]>3.5:
                    predIdx = 1
                elif out[0][2] > out[0][1]:
                    predIdx = 2
                else:
                    predIdx = 0

            signal.pop(0)
            signal.append(predIdx)
            
            if sum(signal) > 5 and flagNothing:
                attemps = attemps + 1
                flagShot = True
                flagNothing = False
            
            if sum(signal) > 8 and flagShot and (not flagMade):
                made = made + 1
                flagMade = True
            
            if sum(signal) < 3 and flagShot:
                if not flagMade:
                    miss = miss + 1
                flagNothing = True
                flagMade = False
                flagShot = False
                display = True


            count = count + 1
            if display:
                print("video second",count/fps)
                print("Shots attemps ",attemps)
                print("Shots Made",made)
                print("Shots Missed",miss)
                print("FG%",(made/attemps)*100)
                display = False

                second = count/fps
                fg = (made/attemps)*100

                dataOuputDict = {"videoSecond": second,
                                "Shots attemps": attemps,
                                "Shots Made": made,
                                "Shots Missed": miss,
                                "FG": fg
                                }

                with open(jsonPath, 'w') as json_file:
                    json.dump(dataOuputDict, json_file)

        # Break the loop
        else:
            break
    cap.release()


def main():
    #parser = argparse.ArgumentParser(description=" nothing")
    #parser.add_argument('--videoPath',type=str,help='path to video file')
    #args = parser.parse_args()

    videoPath = "./4810.mp4"
    modelPath = "./model_example.pt"
    jsonPath = "./output.json"
    hoopCenter = [int(96), int(120)]
    hoopSize = [int(20),int(10)]
    shotPredict(videoPath, modelPath,jsonPath, hoopCenter,hoopSize)
    pass


if __name__ == "__main__":
    main()