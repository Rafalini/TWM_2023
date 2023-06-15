import xml.etree.ElementTree as ET
import json
import shutil
import os
from PIL import Image

annoDirPath = "./annotations/DETRAC-Train-Annotations-XML"
datasetPath = "./Insight-MVT_Annotation_Train"
yoloDataPath = "./yolo/data/"
yoloDataAnnoPath = "./yolo/data/annotations"

if not os.path.exists(yoloDataPath):
    os.makedirs(yoloDataPath)

if not os.path.exists(yoloDataAnnoPath):
    os.makedirs(yoloDataAnnoPath)

def prepareAnnotations():

    for fileName in os.listdir(annoDirPath):
        if os.path.isfile(os.path.join(annoDirPath, fileName)):
            fileNameNoExt = fileName.split('.')[0]

            framesAnnoPath = os.path.join(datasetPath, fileNameNoExt)
            if not os.path.exists(framesAnnoPath):
                os.makedirs(framesAnnoPath)

            tree = ET.parse('annotations/DETRAC-Train-Annotations-XML/' + fileName)
            root = tree.getroot()
            print("processing " + fileName + " ...")
            prefix = fileName.split('.')[0]
            for frame in root:
                if frame.tag == 'frame':
                    objects = []
                    for element in frame[0]:
                        object = {"label": "car"}
                        x_y_w_h = []
                        x_y_w_h.append(float(element[0].attrib['left']))
                        x_y_w_h.append(float(element[0].attrib['top']))
                        x_y_w_h.append(float(element[0].attrib['left']) + float(element[0].attrib['width']))
                        x_y_w_h.append(float(element[0].attrib['top']) - float(element[0].attrib['height']))
                        object["x_y_w_h"] = x_y_w_h
                        objects.append(object)

                    # image_w_h = [960, 540]
                    image_w_h = [640, 480]

                    frameJson = {"objects": objects, "image_w_h": image_w_h}
                    num = f'{int(frame.attrib["num"]):05d}'

                    with open(os.path.join(framesAnnoPath, 'img' + num + '.json'), 'w') as fp:
                        json.dump(frameJson, fp)

                    with open(os.path.join(yoloDataAnnoPath, prefix + '_' + 'img' + num + '.json'), 'w') as fp:
                        json.dump(frameJson, fp)


def copyFiles():
    for dirName in os.listdir(datasetPath):
        for fileName in os.listdir(os.path.join(datasetPath,dirName)):
            # newFileName = dirName +"_"+ fileName
            newFileName = dirName +"_"+ fileName
            print("processing "+newFileName+" ...")
            if fileName.split('.')[1] == "jpg":
            #     shutil.copyfile(os.path.join(datasetPath, dirName, fileName), os.path.join(yoloDataAnnoPath, newFileName))
            # else:
            #     shutil.copyfile(os.path.join(datasetPath, dirName, fileName), os.path.join(yoloDataPath, newFileName))
            # shutil.copyfile(os.path.join(datasetPath, dirName, fileName), os.path.join(yoloDataPath, newFileName))

                image = Image.open(os.path.join(datasetPath, dirName, fileName))
                new_image = image.resize((640, 480))
                new_image.save(os.path.join(yoloDataPath, newFileName))



#
if __name__ == "__main__":
    prepareAnnotations()
    copyFiles()