This is a toy program demonstrating a basic setup for training super-resolution with UNet.
It runs on video data split into individual frame pngs.

Example data setup:
- download a video <video.mp4>
- ffmpeg -ss <start time hh:mm:ss> -to <end time hh:mm:ss> -i <video.mp4> data/%04d.png
- set total_frames, img_width, img_height in Run.py
- set inference_start_frame, inference_frames in Run.py
These parameters can be used to make training / inference data distinct, though by default
I have not bothered and am just letting the network overfit.

You need the CUDA toolkit to run this on the GPU: https://developer.nvidia.com/cuda-downloads
and to install PyTorch with CUDA: https://pytorch.org/get-started/locally/

Otherwise, it will run on CPU. That will be very slow.