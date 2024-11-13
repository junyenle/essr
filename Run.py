from PIL import Image
import random
import subprocess
import torch
import torchmetrics
import torchvision

from Model import UNet, LossFn

# control flow
train = True # else inference
cont = False # if training, continue training last.ckpt

# parameters
in_channels = 6 # this frame's RGB input + previous frame's super-ressed RGB output
learning_rate = 0.001
checkpoint_name = "sr_aloy"
frames_per_clip = 10
total_frames = 2000 
frame_weights = [0.01, 0.04, 0.14, 0.325, 0.6, 0.882, 1, 1, 1, 1] # used to weight loss from frames 
assert(len(frame_weights) >= frames_per_clip)
# NOTE: this program expects the data directory to contain total_frames png frames extracted from video, numbered %04d
# you can do this e.g. with ffmpeg -ss <start time hh:mm:ss> -to <end time> -i <video.mp4> data/%04d.png
data_dir = "./data"
checkpoint_dir = "./checkpoints"
img_width = 1920
img_height = 1080
augment_crop_size = 256
training_steps = 1000000
checkpoint_freq = 1000

# inference parameters
inference_start_frame = 1 # I'm so sorry this is not 0-indexed... blame ffmpeg
inference_frames = 800
inference_checkpoint = "last"
output_dir = "./output"
video_dir = "./videos"
inference_save_input = False
inference_save_output = True
inference_save_target = False


if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

if train:
    print(f"Training on {device}")

    model = UNet(in_channels)
    if cont:
        model.load_state_dict(torch.load(f"{checkpoint_dir}/last.ckpt", weights_only=True))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train() # internal pytorch state important for batch normalization

    for step in range(training_steps):
        if step % 100 == 0:
            print(f"step {step}/{training_steps}")

        inputs = []
        targets = []
        # Extremely primitive data loading
        # pick a random start frame to begin a clip 
        start_frame = random.randint(1, total_frames-frames_per_clip)

        # decide parameters for augmentation
        # these are done here because they need to be applied uniformly across all frames in the clip
        augment_crop_top = random.randint(0, img_height - 1 - augment_crop_size)
        augment_crop_left = random.randint(0, img_width - 1 - augment_crop_size)
        augment_crop_rotate = random.randint(0, 3) * 90

        # build entire clip's input / target tensors
        for offset in range(frames_per_clip):
            frame = start_frame + offset

            # Data is NOT batched, which can lead to slow convergence but is much simpler to understand
            # Data is in c,h,w format. This is industry standard and most libraries will expect it as such.
            img = Image.open(f"{data_dir}/{frame:04d}.png")
            convert_to_tensor = torchvision.transforms.ToTensor()
            target = convert_to_tensor(img)
            # Most libraries expect data to be b,c,h,w but we are using batches of 1. We can just unsqueeze to add the required batch dimension
            target = torch.unsqueeze(target, 0)

            # Augmentation
            target = torchvision.transforms.functional.crop(target, augment_crop_top, augment_crop_left, augment_crop_size, augment_crop_size)
            target = torchvision.transforms.functional.rotate(target, augment_crop_rotate)
            target = target.to(device)

            # NOTE: the input is just the target downsampled.
            input = target.clone().detach()
            input = torch.nn.functional.interpolate(input, size=None, scale_factor=0.5, mode="nearest")
            input = input.to(device)

            inputs.append(input)
            targets.append(target)

        # Create dummy prev_y and prev_target for the first frame
        prev_y = torch.randn(1, 3, augment_crop_size, augment_crop_size).to(device)
        prev_target = torch.randn(1, 3, augment_crop_size, augment_crop_size).to(device)

        for frame in range(frames_per_clip):
            x = inputs[frame]
            target = targets[frame]

            # inference and weight update
            # pre-upscaling
            x = torch.nn.functional.interpolate(x, size=None, scale_factor=2.0, mode="bilinear")

            model_input = torch.cat((x, prev_y), dim=1)
            y = model(model_input)

            loss = frame_weights[frame] * LossFn(y, target, prev_y, prev_target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            prev_y = y.clone().detach()
            prev_target = target.clone().detach()

        if step % checkpoint_freq == 0:
            print(f"Training loss: {loss}")
            checkpoint_filename = f"{checkpoint_dir}/{checkpoint_name}_{step}.ckpt"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint {checkpoint_filename}")
            
            checkpoint_filename = f"{checkpoint_dir}/last.ckpt"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint {checkpoint_filename}")
else:
    # inference
    print(f"Inference on {device}")

    model = UNet(in_channels)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/{inference_checkpoint}.ckpt", weights_only=True))
    model.to(device)
    
    prev_y = torch.randn(1, 3, img_height, img_width).to(device)

    for frame in range(inference_start_frame, inference_start_frame + inference_frames):
        print(f"frame {frame-inference_start_frame}/{inference_frames}")

        img = Image.open(f"{data_dir}/{frame:04d}.png")
        convert_to_tensor = torchvision.transforms.ToTensor()
        target = convert_to_tensor(img)
        target = torch.unsqueeze(target, 0).to(device)
        if inference_save_target:
            torchvision.utils.save_image(target, f"{output_dir}/target_{frame-inference_start_frame:04d}.png")

        input = target.clone().detach()
        input = torch.nn.functional.interpolate(input, size=None, scale_factor=0.5, mode="nearest")
        if inference_save_input:
            torchvision.utils.save_image(input, f"{output_dir}/in_{frame-inference_start_frame:04d}.png")

        if inference_save_output:
            input = torch.nn.functional.interpolate(input, size=None, scale_factor=2.0, mode="nearest")
            input = input.to(device)
            model_input = torch.cat((input, prev_y), dim=1)
            y = model(model_input)
            prev_y = y.clone().detach()
            torchvision.utils.save_image(y, f"{output_dir}/out_{frame-inference_start_frame:04d}.png")

    if inference_save_input:
        ffmpeg_cmd = f"ffmpeg -framerate 60 -apply_trc gamma22  -y -i {output_dir}/in_%4d.png -vcodec libx264 {video_dir}/in.mp4"
        subprocess.call(ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL)
    if inference_save_output:
        ffmpeg_cmd = f"ffmpeg -framerate 60 -apply_trc gamma22  -y -i {output_dir}/out_%4d.png -vcodec libx264 {video_dir}/out.mp4"
        subprocess.call(ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL)
    if inference_save_target:
        ffmpeg_cmd = f"ffmpeg -framerate 60 -apply_trc gamma22  -y -i {output_dir}/target_%4d.png -vcodec libx264 {video_dir}/target.mp4"
        subprocess.call(ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL)