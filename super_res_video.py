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

ret, frame = cap.read()

upscaled = sr.upsample(frame)
frame_width = int(upscaled.shape[1])
frame_height = int(upscaled.shape[0])

# Write the frame into the file
# Define the codec and create VideoWriter object.The output is stored in a file.
out = cv2.VideoWriter('output/output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30,
                      (frame_width, frame_height))

while cap.isOpened():
    # catpure frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        upscaled = sr.upsample(frame)
        out.write(upscaled)
        print(f"{frame.shape[1]} {frame.shape[0]} -> {upscaled.shape[1]} {upscaled.shape[0]}")

    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
