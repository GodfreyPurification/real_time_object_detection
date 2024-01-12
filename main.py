# https://www.youtube.com/watch?v=hVavSe60M3g
# https://github.com/haroonshakeel/Real_Time_object_detection_CPU
from Detector import* 
import os 

def main():
    # videoPath ="D://python//nms//test_video//video.mp4"
    videoPath ="D://python//nms//test_video//pexels_videos_1721294.mp4"

    configPath=os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath=os.path.join("model_data","frozen_inference_graph.pb")
    classPath=os.path.join("model_data","coco.names")
    
    detector=Detector(videoPath, configPath, modelPath, classPath)
    detector.onVideo()
if __name__=='__main__':
    main()