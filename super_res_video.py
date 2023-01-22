"""
python super_res_video.py --model models/ESPCN_x4.pb --video examples/sample.mp4
"""
import argparse
import time
import cv2
import os

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m",
    "--model",
    required=True,
    help="path to super resolution model"
)
ap.add_argument(
    "-i",
    "--video",
    required=True,
    help="path to input video clip we want to increase resolution of"
)
args = vars(ap.parse_args())
input_video = args['video']
video_name, input_video_ext = args["video"].split(os.path.sep)[-1].split('.')

# extract the model name and model scale from the file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print(f"[INFO] loading super resolution model: {args['model']}")
print(f"[INFO] model name: {modelName}")
print(f"[INFO] model scale: {modelScale}")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

cap = cv2.VideoCapture(input_video)
out = None
while cap.isOpened():
    # catpure frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        upscaled = sr.upsample(frame)

        frameWidth = upscaled.shape[1]
        frameHeight = upscaled.shape[0]
        # Write the frame into the file
        # Define the codec and create VideoWriter object.The output is stored in a file.
        out = cv2.VideoWriter('output/output.avi',
                              cv2.VideoWriter_fourcc(*"MJPG"),
                              30,
                              (frameWidth, frameHeight))
        out.write(upscaled)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
