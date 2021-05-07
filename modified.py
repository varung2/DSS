import config
import torch

from DSS.core.cloud import PointClouds3D

if __name__ == '__main__':
	# note this only works for cuda
	# is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda")

	# testing the splatting renderer
	splatting_renderer = config.create_splatting_renderer().to(device)
	print('Splatting renderer type: {}'.format(type(splatting_renderer)))

	# generating 1 batche(s) of 1000 points with 10 features for each point
	batch_size = 1
	rand_pts_pos = torch.randn((batch_size, 1000, 3), device=device);
	rand_pts_normals = torch.randn((batch_size, 1000, 3), device=device);
	rand_pts_feat = torch.randn((batch_size, 1000, 10), device=device);
	print('DEBUG: rand_pts_pos shape: {}'.format(rand_pts_pos.shape))
	print('DEBUG: rand_pts_feat shape: {}'.format(rand_pts_feat.shape))

	pt_cloud = PointClouds3D(points=rand_pts_pos, normals=rand_pts_normals, features=rand_pts_feat)
	if (batch_size > 1): pt_cloud.extend(batch_size)
	print('Point cloud: {}'.format(pt_cloud))

	# checking output images shape and channel size
	images = splatting_renderer(pt_cloud)
	print('Images type: {}'.format(type(images)))
	print('Images shape: {}'.format(images.shape))