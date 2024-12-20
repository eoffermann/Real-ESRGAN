import gradio as gr
import os
import mimetypes
import torch
import subprocess
from tqdm import tqdm
from os import path as osp
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

# Supporting classes and functions (Reader, Writer) would be added here, as in the original script

class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def upscale_video(
    input_path,
    model_name='realesr-animevideov3',
    output_folder='results',
    denoise_strength=0.5,
    outscale=4,
    suffix='out',
    tile=0,
    tile_pad=10,
    pre_pad=0,
    face_enhance=False,
    fps=None,
    ffmpeg_bin='ffmpeg',
    extract_frame_first=False):
    """Processes a video using Real-ESRGAN and returns the output path."""

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Determine input type
    input_path = input_path.rstrip('/').rstrip('\\')
    input_type = mimetypes.guess_type(input_path)[0]
    is_video = input_type is not None and input_type.startswith('video')

    # Handle .flv case
    if is_video and input_path.endswith('.flv'):
        mp4_path = input_path.replace('.flv', '.mp4')
        os.system(f'{ffmpeg_bin} -i {input_path} -codec copy {mp4_path}')
        input_path = mp4_path

    if extract_frame_first and not is_video:
        extract_frame_first = False

    # Video naming and save path
    video_name = osp.splitext(os.path.basename(input_path))[0]
    video_save_path = osp.join(output_folder, f'{video_name}_{suffix}.mp4')

    # Select model and initialize upsampler
    if model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
    else:
        # Other model cases would be handled here...
        pass

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=True,  # Use half precision by default
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Handle face enhancement
    if 'anime' in model_name and face_enhance:
        face_enhance = False  # Face enhancement not supported for anime models

    face_enhancer = None
    if face_enhance:
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    # Process video frames
    reader = Reader(input_path=input_path, ...)
    writer = Writer(output_folder=output_folder, ...)

    pbar = tqdm(total=len(reader), unit='frame', desc='Inference')
    while True:
        img = reader.get_frame()
        if img is None:
            break

        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print(f'Error during processing: {error}')
        else:
            writer.write_frame(output)

        pbar.update(1)

    reader.close()
    writer.close()

    return video_save_path

# Gradio Interface
def gradio_interface(input_path, model_name, output_folder, denoise_strength, outscale, suffix, tile, tile_pad, pre_pad, face_enhance, fps, extract_frame_first):
    output = upscale_video(input_path, model_name, output_folder, denoise_strength, outscale, suffix, tile, tile_pad, pre_pad, face_enhance, fps, extract_frame_first=extract_frame_first)
    return f"Processing complete. Output saved at {output}"

# Define Gradio inputs and outputs
inputs = [
    gr.Textbox(label="Input Path", placeholder="Path to input video/image folder"),
    gr.Dropdown(choices=["realesr-animevideov3", "RealESRGAN_x4plus"], label="Model Name"),
    gr.Textbox(label="Output Folder", placeholder="Path to save output"),
    gr.Slider(0.0, 1.0, value=0.5, label="Denoise Strength"),
    gr.Slider(1, 4, value=4, label="Outscale"),
    gr.Textbox(label="Suffix", value="out"),
    gr.Slider(0, 256, value=0, step=16, label="Tile Size"),
    gr.Slider(0, 32, value=10, step=2, label="Tile Padding"),
    gr.Slider(0, 16, value=0, step=2, label="Pre Padding"),
    gr.Checkbox(label="Face Enhance"),
    gr.Number(label="FPS (Optional)", value=None),
    gr.Checkbox(label="Extract Frames First"),
]

outputs = gr.Textbox(label="Output Log")

gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=outputs,
    title="Real-ESRGAN Video Upscaler"
).launch()
