import config
import torch

from DSS.core.cloud import PointClouds3D

if __name__ == '__main__':

	# testing the splatting renderer
	splatting_renderer = config.create_splatting_renderer()
	print('Splatting renderer type: {}'.format(type(splatting_renderer)))

	rand_pts_pos = torch.randn((1, 1000, 3));
	rand_pts_feat = torch.randn((1, 1000, 10));
	print('DEBUG: rand_pts_pos shape: {}'.format(rand_pts_pos.shape))
	print('DEBUG: rand_pts_feat shape: {}'.format(rand_pts_feat.shape))

	pt_cloud = PointClouds3D(points=rand_pts_pos, features=rand_pts_feat)
	print('Point cloud type: {}'.format(type(pt_cloud)))

	images = splatting_renderer(pt_cloud)
	print('Images type: {}'.format(type(images)))
	print('Images shape: {}'.format(images.shape))