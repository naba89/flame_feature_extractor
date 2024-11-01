import torch

from .flame import FLAME
from .renderer_utils import RenderMesh


class FlameRenderer:
    def __init__(
        self,
        fixed_transform=False,
        device="cuda",
        max_batch_size: int = 128,
        n_shape: int = 100,
        n_exp: int = 50,
        scale: float = 5.0,
        render_size: int = 512,
    ):

        self.device = device
        self.max_batch_size = max_batch_size
        self.fixed_transform = fixed_transform
        self.flame_model = FLAME(n_shape=n_shape, n_exp=n_exp, scale=scale).to(device)
        self.mesh_render = RenderMesh(render_size, faces=self.flame_model.get_faces().cpu().numpy(), device=device)
        self.transform_matrix = torch.tensor(
            [[[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 12.0]]], dtype=torch.float32, device=device
        )

    def render(self, shape, expression, pose, eye=None, transform_matrix=None, **kwargs):
        if type(shape) != torch.Tensor:
            shape = torch.tensor(shape, dtype=torch.float32, device=self.device)[None]
            expression = torch.tensor(expression, dtype=torch.float32, device=self.device)[None]
            pose = torch.tensor(pose, dtype=torch.float32, device=self.device)[None]
        else:
            shape = shape.to(self.device)[None]
            expression = expression.to(self.device)[None]
            pose = pose.to(self.device)[None]

        vertices, _ = self.flame_model(
            shape_params=shape, expression_params=expression, pose_params=pose, eye_pose_params=None
        )
        if transform_matrix is None or self.fixed_transform:
            transform_matrix = self.transform_matrix
        elif type(transform_matrix) != torch.Tensor:
            transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=self.device)[None]
        else:
            transform_matrix = transform_matrix.to(self.device)[None]
        images, alpha_images = self.mesh_render(
            vertices,
            focal_length=12.0,
            transform_matrix=transform_matrix,
        )
        return images, alpha_images

    def render_batch(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        pose: torch.Tensor,
        transform_matrix: torch.Tensor | None = None,
    ):
        bs = shape.size(0)
        vertices, _ = self.flame_model(
            shape_params=shape,
            expression_params=expression,
            pose_params=pose,
            eye_pose_params=None,
        )
        if transform_matrix is None or self.fixed_transform:
            transform_matrix = self.transform_matrix
        elif type(transform_matrix) != torch.Tensor:
            transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=self.device)[None]
        else:
            transform_matrix = transform_matrix.to(self.device)[None]

        images, alpha_images = [], []
        for i in range(0, bs, self.max_batch_size):
            ending_index = min(bs, i + self.max_batch_size)
            image, alpha_image = self.mesh_render(
                vertices[i:ending_index],
                focal_length=12.0,
                transform_matrix=transform_matrix,
            )
            images.append(image)
            alpha_images.append(alpha_image)
        return torch.cat(images, dim=0), torch.cat(alpha_images, dim=0)
