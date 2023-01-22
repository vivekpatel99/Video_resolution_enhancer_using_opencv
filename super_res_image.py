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
    "--image",
    required=True,
    help="path to input image we want to increase resolution of"
)
args = vars(ap.parse_args())

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

# load the input image from disk and display its spatial dimension
image = cv2.imread(args["image"])
print(f"[INFO] w:{image.shape[1]}, h: {image.shape[0]}")

# use the super resolution model to upscale the image, timing now
# long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print(f"[INFO] super resolution took {end - start:.6f}")

# show the spatial dimension of the super resolution image
print(f"[INFO] w:{upscaled.shape[1]}, h: {upscaled.shape[0]}")

# resize the image using standard bicubic interpolation
start = time.time()
bicubic = cv2.resize(image,
                     (upscaled.shape[1],
                      upscaled.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
end = time.time()
print(f"[INFO] bicubic interpolation took {end - start:.6f}")

# show the original input image, bicubic interpolation image, and
# super resolution deep learning output
cv2.imshow("Original", image)
cv2.imshow("Bicubic", bicubic)
cv2.imshow("Super Resolution", upscaled)

imageName = args["image"].split(os.path.sep)[-1].split('.')[0]
cv2.imwrite(f"output/{imageName}_high.png", upscaled)

cv2.waitKey(0)
