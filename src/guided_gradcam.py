import torch
from misc_functions import *
from guided_backprop import *
from gradcam import *

class Guided_GradCam():
	def __init__(self, model, target_layer):
		self.model = model
		self.model.eval()
		self.gradcam = GradCam(self.model, target_layer)
		self.guide = GuidedBackprop(self.model)

	def generate_guided_grad(self, input_image, class_label=None):
		cam = self.gradcam.generate_cam(input_image, class_label)
		grad = self.guide.generate_gradients(input_image, class_label)
		return cam*grad

if __name__ == '__main__':
	# Get params
	target_example = 2  # Snake
	(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
		get_example_params(target_example)

	guide_gradcam = Guided_GradCam(pretrained_model, target_layer=29)
	cam_gb = guide_gradcam.generate_guided_grad(prep_img, target_class)
	save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
	grayscale_cam_gb = convert_to_grayscale(cam_gb)
	save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
	print('Guided grad cam completed')
