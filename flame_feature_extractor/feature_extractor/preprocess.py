import cv2
import numpy as np
import torch
import torchvision
from batch_face import RetinaFace
import mediapipe

from .mica.insightface import face_align


class PreProcessBatchFace:
    def __init__(self, gpu_id=-1):
        self.detector = RetinaFace(fp16=False, gpu_id=gpu_id)
        self.threshold = 0.95

    def __call__(self, frames):
        """

        :param frames: RGB images of shape (B, 3, H, W), torch tensor
        :return:
        """
        frames = frames.permute(0, 2, 3, 1)  # (B, H, W, 3)

        all_faces = self.detector(frames.cpu().numpy(), threshold=self.threshold)
        frames = frames.permute(0, 3, 1, 2)  # (B, 3, H, W)
        mica_images = []
        emoca_images = []
        for frame, face in zip(frames, all_faces):
            box, kps, score = face[0]  # the first face's detection result

            aimg = face_align.norm_crop(frame.permute(1, 2, 0).cpu().numpy(), landmark=kps)
            blob = cv2.dnn.blobFromImages([aimg], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            mica_image = torch.tensor(blob[0])
            mica_images.append(mica_image)

            size = int((box[2] + box[3] - box[0] - box[1]) / 2 * 1.25)
            center = torch.tensor([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            emoca_image = torchvision.transforms.functional.crop(
                frame.float(),
                top=int(center[1] - size / 2), left=int(center[0] - size / 2),
                height=size, width=size,
            )
            emoca_images.append(emoca_image)

        mica_images = torch.stack(mica_images)
        emoca_images = torch.stack(emoca_images)
        emoca_images = torchvision.transforms.functional.resize(emoca_images, size=224, antialias=True) / 255.0

        return {'mica_images': mica_images, 'emoca_images': emoca_images}


class PreProcessMediaPipe:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,):

        self.dense_lmks_model = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces, refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def __call__(self, frames):
        """

        :param frames: RGB images of shape (B, 3, H, W), torch tensor
        :return:
        """
        # frames = frames.permute(0, 2, 3, 1)  # (B, H, W, 3)
        emoca_images = []
        for frame in frames:
            lmk_image = np.transpose(frame.cpu().numpy(), (1, 2, 0))
            lmks_dense = self.dense_lmks_model.process(lmk_image)
            if lmks_dense.multi_face_landmarks is None:
                return None
            else:
                lmks_dense = lmks_dense.multi_face_landmarks[0].landmark
                # lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
                lmks_dense = np.array([[l.x, l.y] for l in lmks_dense])
                lmks_dense[:, 0] = lmks_dense[:, 0] * lmk_image.shape[1]
                lmks_dense[:, 1] = lmks_dense[:, 1] * lmk_image.shape[0]
                lmks_dense = torch.tensor(lmks_dense)

            min_xy = lmks_dense.min(dim=0)[0]
            max_xy = lmks_dense.max(dim=0)[0]
            box = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
            size = int((box[2] + box[3] - box[0] - box[1]) / 2 * 1.25)
            center = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

            emoca_image = torchvision.transforms.functional.crop(
                frame.float(),
                top=int(center[1] - size / 2), left=int(center[0] - size / 2),
                height=size, width=size,
            )
            emoca_image = torchvision.transforms.functional.resize(emoca_image, size=224, antialias=True) / 255.0
            emoca_images.append(emoca_image)

        emoca_images = torch.stack(emoca_images)

        return {'emoca_images': emoca_images}



class PreProcessMediaPipeDense:
    def __init__(self):

        self.dense_lmks_model = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def __call__(self, frames):
        """

        :param frames: RGB images of shape (B, 3, H, W), torch tensor
        :return:
        """
        # frames = frames.permute(0, 2, 3, 1)  # (B, H, W, 3)
        emoca_images = []
        lmks_denses = []
        for frame in frames:
            lmk_image = np.transpose(frame.cpu().numpy(), (1, 2, 0))
            lmks_dense = self.dense_lmks_model.process(lmk_image)
            if lmks_dense.multi_face_landmarks is None:
                return None
            else:
                lmks_dense = lmks_dense.multi_face_landmarks[0].landmark
                # lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
                lmks_dense = np.array([[l.x, l.y] for l in lmks_dense])
                lmks_dense[:, 0] = lmks_dense[:, 0] * lmk_image.shape[1]
                lmks_dense[:, 1] = lmks_dense[:, 1] * lmk_image.shape[0]
                lmks_dense = torch.tensor(lmks_dense)

            min_xy = lmks_dense.min(dim=0)[0]
            max_xy = lmks_dense.max(dim=0)[0]
            box = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
            size = int((box[2] + box[3] - box[0] - box[1]) / 2 * 1.25)
            center = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

            emoca_image = torchvision.transforms.functional.crop(
                frame.float(),
                top=int(center[1] - size / 2), left=int(center[0] - size / 2),
                height=size, width=size,
            )
            emoca_image = torchvision.transforms.functional.resize(emoca_image, size=224, antialias=True) / 255.0
            emoca_images.append(emoca_image)
            lmks_denses.append(lmks_dense)

        emoca_images = torch.stack(emoca_images)
        lmks_denses = torch.stack(lmks_denses)

        return {'emoca_images': emoca_images, 'lmks_dense': lmks_denses}
