import cv2
import numpy as np 
import time
np.random.seed(20)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classPath):
        self.videoPath=videoPath
        self.configPath=configPath
        self.modelPath=modelPath 
        self.classPath=classPath
        ####################################################3
        # initialize the network
        self.net=cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        #inage scale -1 to 1
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        
    
    def readClasses(self):
        with open(self.classPath,'r') as f:
            self.classesList=f.read().splitlines()

        self.classesList.insert(0,'__Background__')
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3))


        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening file")
            return
        startTime=0
        while True:
            success, image = cap.read()
            currentTime=time.time()
            fps=1/(currentTime-startTime)
            startTime=currentTime

            if not success:
                break
            desired_width = 640
            aspect_ratio = image.shape[1] / image.shape[0]
            desired_height = int(desired_width / aspect_ratio)

            image = cv2.resize(image, (desired_width, desired_height))
            classLabelIDs, confidences, bboxes = self.net.detect(image, confThreshold=0.5)
            bboxes = list(bboxes)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            bboxIdx = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.5, nms_threshold=0.2)



            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxes[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor=[int(c) for c in self.colorList[classLabelID]]
                    displayText= "{}:{:.2f}".format(classLabel,classConfidence)

                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,classColor,2)
                    ####################################################################################3333
                    lineWidth = min(int(w * 0.3),int(h * 0.3))
                    cv2.line(image,(x,y),(x+lineWidth,y),classColor,thickness=5)
                    cv2.line(image, (x, y), (x, y+lineWidth), classColor, thickness=5)
                    cv2.line(image, (x+w, y), (x+w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x+ w, y), (x+w, y + lineWidth), classColor, thickness=5)
                    #####################################################################
                    cv2.line(image, (x, y+h), (x + lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x, y+h), (x, y +h- lineWidth), classColor, thickness=5)
                    cv2.line(image, (x + w, y+h), (x + w - lineWidth, y+h), classColor, thickness=5)
                    cv2.line(image, (x+ w , y+h), (x+w, y  +h-lineWidth), classColor, thickness=5)
                cv2.putText(image,"FPS: " +str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                cv2.imshow("Result", image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()




