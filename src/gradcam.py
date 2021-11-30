import torch
from misc_functions import *
from PIL import Image
import numpy as np 

class CamExtractor():
	def __init__(self, model, target_layer):
		self.model = model
		self.target_layer = target_layer
		self.gradients = None

	def save_grad(self, grad):
		self.gradients = grad

	def forward_pass(self, x):
		conv_output = None
		for module_pos, module in self.model.features._modules.items():
			x = module(x)
			if int(module_pos) == self.target_layer:
				x.register_hook(self.save_grad)
				conv_output = x
		x = x.view(x.size(0),-1)
		x = self.model.classifier(x)
		return conv_output, x

class GradCam():
	def __init__(self, model, target_layer):
		self.model = model
		self.model.eval()
		self.extractor = CamExtractor(model, target_layer)

	def generate_cam(self, input_image, class_label=None):
		conv_output, model_output = self.extractor.forward_pass(input_image)

		if class_label is None:
			class_label = np.argmax(model_output.data.numpy())

		one_hot_output = torch.zeros((1,model_output.size()[-1]))
		one_hot_output[0][class_label] = 1
		print(model_output[0][class_label])

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()

		model_output.backward(gradient=one_hot_output, retain_graph=True)
		guided_gradients = self.extractor.gradients.data.numpy()[0]
		target = conv_output.data.numpy()[0]
		weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
		# cam = np.ones(target.shape[1:], dtype=np.float32)
		cam = np.zeros(target.shape[1:], dtype=np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]
		cam = np.maximum(cam, 0)
		cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
		cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
		cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
					   input_image.shape[3]), Image.ANTIALIAS))/255

		return cam

if __name__ == '__main__':
	target_example = 7  # Snake
	(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
		get_example_params(target_example)
	# Grad cam
	grad_cam = GradCam(pretrained_model, target_layer=29)
	# Generate cam mask
	cam = grad_cam.generate_cam(prep_img, target_class)
	# Save mask
	save_class_activation_images(original_image, cam, file_name_to_export)
	print('Grad cam completed')