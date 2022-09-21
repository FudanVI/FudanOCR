
from __future__ import absolute_import

import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

def grid_sample(input, grid, canvas = None):
  output = F.grid_sample(input, grid)
  if canvas is None:
    return output
  else:
    input_mask = input.data.new(input.size()).fill_(1)
    output_mask = F.grid_sample(input_mask, grid)
    padded_output = output * output_mask + canvas * (1 - output_mask)
    return padded_output


def compute_partial_repr(input_points, control_points):
  N = input_points.size(0)
  M = control_points.size(0)
  pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
  pairwise_diff_square = pairwise_diff * pairwise_diff
  pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
  repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
  mask = repr_matrix != repr_matrix
  repr_matrix.masked_fill_(mask, 0)
  return repr_matrix


def build_output_control_points(num_control_points, margins):
  margin_x, margin_y = margins
  num_ctrl_pts_per_side = num_control_points // 2
  ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
  ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
  ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
  ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
  ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)

  output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
  output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
  return output_ctrl_pts



class TPSSpatialTransformer(nn.Module):

  def __init__(self, output_image_size=None, num_control_points=None, margins=None):
    super(TPSSpatialTransformer, self).__init__()
    self.output_image_size = output_image_size
    self.num_control_points = num_control_points
    self.margins = margins

    self.target_height, self.target_width = output_image_size
    target_control_points = build_output_control_points(num_control_points, margins)
    N = num_control_points

    forward_kernel = torch.zeros(N + 3, N + 3)
    target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
    forward_kernel[:N, :N].copy_(target_control_partial_repr)
    forward_kernel[:N, -3].fill_(1)
    forward_kernel[-3, :N].fill_(1)
    forward_kernel[:N, -2:].copy_(target_control_points)
    forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

    inverse_kernel = torch.inverse(forward_kernel)

    HW = self.target_height * self.target_width
    target_coordinate = list(itertools.product(range(self.target_height), range(self.target_width)))
    target_coordinate = torch.Tensor(target_coordinate)
    Y, X = target_coordinate.split(1, dim=1)
    Y = Y / (self.target_height - 1)
    X = X / (self.target_width - 1)
    target_coordinate = torch.cat([X, Y], dim=1)
    target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
    target_coordinate_repr = torch.cat([
      target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
    ], dim = 1)

    self.register_buffer('inverse_kernel', inverse_kernel)
    self.register_buffer('padding_matrix', torch.zeros(3, 2))
    self.register_buffer('target_coordinate_repr', target_coordinate_repr)
    self.register_buffer('target_control_points', target_control_points)

  def forward(self, input, source_control_points):
    assert source_control_points.ndimension() == 3
    assert source_control_points.size(1) == self.num_control_points
    assert source_control_points.size(2) == 2
    batch_size = source_control_points.size(0)

    Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
    mapping_matrix = torch.matmul(self.inverse_kernel, Y)
    source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)

    grid = source_coordinate.view(-1, self.target_height, self.target_width, 2)
    grid = torch.clamp(grid, 0, 1)
    grid = 2.0 * grid - 1.0
    output_maps = grid_sample(input, grid, canvas=None)
    return output_maps, source_coordinate