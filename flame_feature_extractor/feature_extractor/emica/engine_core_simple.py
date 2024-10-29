import os
import shutil
import time

import torch
import torchvision
import torchvision.transforms.functional as tf

from engines.engine_core import EmicaCore, expand_bbox, data_to_device


class EmicaExtractionEngine(EmicaCore):
    def __init__(self, focal_length, device="cuda"):
        super().__init__(focal_length=focal_length, device=device)

    @staticmethod
    def get_video_name_and_prepare_path(video_path: str, output_path: str):
        video_name = os.path.basename(video_path).split(".")[0]
        if os.path.exists(output_path):
            print(f"Output path {output_path} exists, replace it.")
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        return video_name

    def extract_optim_flame(self, video_path: str, output_path: str, device_id: int):
        start = time.time()
        video_name = self.get_video_name_and_prepare_path(video_path, output_path)

        frames_data, audio_data, meta_data = torchvision.io.read_video(video_path, output_format="TCHW")
        fps = int(meta_data["video_fps"])
        assert frames_data.shape[0] > 0, "No frames in the video, reading video failed."
        # print(f"Processing video {video_path} with {frames_data.shape[0]} frames on device: {device_id}.")

        all_frames_boxes, all_frames_idx = [], []
        for fidx, frame in enumerate(frames_data):
            _, bbox, _ = self.vgghead_encoder(frame, fidx, only_vgghead=True)
            if bbox is not None:
                all_frames_idx.append(fidx)
                all_frames_boxes.append(bbox.cpu())
        if len(all_frames_boxes) < (len(frames_data) * 0.75):
            print(
                f"Video has {len(frames_data)} frames but only detected {len(all_frames_boxes)} head, less than 75%, skipping video"
            )
            return None
        # print(f"{video_path} face extraction complete on device: {device_id}, extracting flame feature...")
        frames_data = frames_data[all_frames_idx]
        all_frames_boxes = self.smooth_bbox(all_frames_boxes, alpha=0.03)
        results = []
        for fidx, frame in enumerate(frames_data):
            frame_bbox = all_frames_boxes[fidx]
            frame_bbox = expand_bbox(frame_bbox, scale=1.65).long()
            crop_frame = tf.crop(
                frame,
                top=frame_bbox[1],
                left=frame_bbox[0],
                height=frame_bbox[3] - frame_bbox[1],
                width=frame_bbox[2] - frame_bbox[0],
            )
            crop_frame = tf.resize(crop_frame, (512, 512), antialias=True)
            # frame = tf.center_crop(frame, 512)
            image = self.matting_engine(crop_frame / 255.0, return_type="matting", background_rgb=0.0).cpu() * 255.0
            image_key = f"{video_name}_{fidx}"
            emica_inputs = self.emica_data_engine(image, image_key)
            emica_inputs = torch.utils.data.default_collate([emica_inputs])
            emica_inputs = data_to_device(emica_inputs, device=self._device)
            emica_results = self.emica_encoder(emica_inputs)
            results.append(
                {
                    "shapecode": emica_results["shapecode"][0].cpu(),
                    "expcode": emica_results["expcode"][0].cpu(),
                    "globalpose": emica_results["globalpose"][0].cpu(),
                    "jawpose": emica_results["jawpose"][0].cpu(),
                    "fps": fps,
                }
            )
        duration = time.time() - start
        fps = duration / frames_data.shape[0]
        print(
            f"{video_path} with {frames_data.shape[0]} frames, processing complete on device: {device_id}, cost {duration:.2f} second, FPS: {fps:.1f}"
        )
        return results
