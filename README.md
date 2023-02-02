# Video_resolution_enhancer_using_opencv

The script will convert low resolution images and video to high resolution images and videos with help of pretrainned deep learning models.

Used deep learning models
* EDSR: [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921),  [implementation](https://github.com/Saafke/EDSR_Tensorflow)
* ESPCN: [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158),  [implementation](https://github.com/fannymonori/TF-ESPCN)
* FSRCNN: [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367),  [implementation](https://github.com/Saafke/FSRCNN_Tensorflow)
* LapSRN: [Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks](https://arxiv.org/abs/1710.01992),  [implementation](https://github.com/fannymonori/TF-LAPSRN)

## HowTo use the script
1. clone the repo
2. add input images or video in examples dir
3. check models in model dir, if you want to add other model then you download it from github repo of pretrained model
4. run following command to run the script
   for Image \
    `python super_res_image.py --model models/<modelname>.pb  --image examples/<imagename>`  \
   for video \
   `python super_res_video.py --model models/<modelname>.pb --video examples/<video_clip_name>` 
5. Generated output will be in output dir


## Future Ideas
* Add audio from sample/input video clip to output video file to complete the whole project. 

## Reference
* https://pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/
* https://bleedai.com/super-resolution-going-from-3x-to-8x-resolution-in-opencv/

