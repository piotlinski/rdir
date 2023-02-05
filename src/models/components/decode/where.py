"""DIR transformer for reconstructed objects."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhereTransformer(nn.Module):
    """Transforms decoded objects using where boxes."""

    def __init__(self, image_size: int, inverse: bool = False, square: bool = False):
        """
        :param image_size: size of transformed object image
        :param inverse: apply inverse transformation
        """
        super().__init__()

        self.image_size = image_size
        self.inverse = inverse
        self.square = square

    @staticmethod
    def convert_to_square(z_where: torch.Tensor) -> torch.Tensor:
        """Make rectangular boxes square."""
        wh = (
            (torch.argmax(z_where[..., 2:], dim=-1) + 2)
            .unsqueeze(-1)
            .expand(*z_where.shape[:-1], 2)
        )
        xy = wh.new_tensor([0, 1]).expand_as(wh)
        index = torch.cat([xy, wh], dim=-1)
        return torch.gather(z_where, -1, index=index)

    @staticmethod
    def scale_boxes(where_boxes: torch.Tensor) -> torch.Tensor:
        """Adjust scaled XYWH boxes to STN format.

        .. t_{XY} = (1 - 2 * {XY}) * s_{WH}
           s_{WH} = 1 / {WH}
        :param where_boxes: latent - detection box
        :return: scaled box
        """
        xy = where_boxes[..., :2]
        wh = where_boxes[..., 2:]
        scaled_wh = 1 / wh
        scaled_xy = (1 - 2 * xy) * scaled_wh
        return torch.cat((scaled_xy, scaled_wh), dim=-1)

    @staticmethod
    def convert_boxes_to_theta(where_boxes: torch.Tensor) -> torch.Tensor:
        """Convert where latents to transformation matrix.

        .. [ w_scale    0    x_translation ]
           [    0    h_scale y_translation ]
        :param where_boxes: latent - detection box
        :return: transformation matrix for transposing and scaling
        """
        n_boxes = where_boxes.shape[0]
        transformation_mtx = torch.cat(
            (torch.zeros((n_boxes, 1), device=where_boxes.device), where_boxes), dim=1
        )
        return transformation_mtx.index_select(
            dim=1,
            index=torch.tensor([3, 0, 1, 0, 4, 2], device=where_boxes.device),
        ).view(n_boxes, 2, 3)

    @staticmethod
    def get_inverse_theta(theta: torch.Tensor) -> torch.Tensor:
        """Get inverse transformation matrix.

        :param theta: transformation matrix for transposing and scaling
        :return: inverted transformation matrix
        """
        last_row = theta.new_tensor([0.0, 0.0, 1.0]).expand(theta.shape[0], 1, 3)
        transformation_mtx = torch.cat((theta, last_row), dim=1)
        return transformation_mtx.inverse()[:, :-1]

    def forward(
        self, decoded_images: torch.Tensor, where_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Takes decoded images (sum_features(grid*grid) x 3 x 64 x 64)

        .. and bounding box parameters x_center, y_center, w, h tensor ..
        (sum_features(grid*grid*n_boxes) x 4) .. and outputs transformed images
        .. (sum_features(grid*grid*n_boxes) x 3 x image_size x image_size)
        """
        n_objects = decoded_images.shape[0]
        channels = decoded_images.shape[1]
        if where_boxes.numel():
            if self.square:
                where_boxes = self.convert_to_square(where_boxes)
            scaled_boxes = self.scale_boxes(where_boxes)
            theta = self.convert_boxes_to_theta(where_boxes=scaled_boxes)
            if self.inverse:
                theta = self.get_inverse_theta(theta)
            grid = F.affine_grid(
                theta=theta,
                size=[n_objects, channels, self.image_size, self.image_size],
            )
            transformed_images = F.grid_sample(input=decoded_images, grid=grid)
        else:
            transformed_images = decoded_images.view(
                -1, channels, self.image_size, self.image_size
            )
        return torch.clamp(transformed_images, 0.0, 1.0)
