import torch
from misc_functions import *

class AttackFGS:

	def __init__(
			self,
			targeted=True, max_epsilon=0.21, min_epsilon=1e-10, norm=2,  # also supports 'inf'
			optimize_epsilon=True, num_iter=None, cuda=True, debug=False):

		self.targeted = targeted
		# self.eps = math.exp((math.log(min_epsilon) + math.log(max_epsilon))/2)
		self.eps = max_epsilon
		self.norm = float(norm)
		self.optimize_epsilon = optimize_epsilon
		if self.optimize_epsilon:
			self.num_iter = num_iter or 10

		self.loss_fn = torch.nn.CrossEntropyLoss()
		if cuda:
			self.loss_fn = self.loss_fn.cuda()
		self.debug = debug

	def generate_ad_ex(self, model, inputs, labels, targets=None, batch_num=0):  # Targets are not one hot encoded
		# input_var = Variable(input, requires_grad=True)
		# target_var = Variable(target)
		eps = self.eps

		if self.optimize_epsilon:
			raise NotImplementedError
		else:
			outputs = model(inputs)

			if outputs.is_cuda:
				idxout = torch.LongTensor(range(outputs.size()[0])).cuda()
				xones = torch.ones(outputs.size()[0]).cuda()
			else:
				idxout = torch.LongTensor(range(outputs.size()[0]))
				xones = torch.ones(outputs.size()[0])

			if self.targeted:
				assert targets is not None, "Please pass in targets for a targeted attack!"

				attack_loss = outputs[idxout, targets]
			else:
				# negative sign below gives us correct direction for attack
				attack_loss = -1*outputs[idxout, labels]

			attack_loss.backward(xones)

			grad_sign = torch.sign(inputs.grad)

			delta = eps*grad_sign

			ad_inputs = inputs + delta

		return ad_inputs

if __name__ == '__main__':
	target_example = 2  # Snake
	(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
		get_example_params(target_example)
	fgs = AttackFGS(targeted=False, optimize_epsilon=False, max_epsilon=0.6)
	orig_output = pretrained_model(prep_img)
	_, orig_pred = torch.max(orig_output.data,1)
	print("Original Class Prediction: {}".format(orig_pred.item()))

	perturbed_image = fgs.generate_ad_ex(pretrained_model, prep_img, orig_pred)

	perturbed_output = pretrained_model(perturbed_image)
	_, adv_pred = torch.max(perturbed_output.data, 1)
	print("Perturbed class prediction: {}".format(adv_pred.item()))

	recreated_perturbed = recreate_image(perturbed_image)
	recreated_perturbed = Image.fromarray(recreated_perturbed.astype("uint8"))

	exmp = ['snake', 'cat_dog', 'spider']
	save_image(recreated_perturbed,'../input_images/adv_'+exmp[target_example]+'.jpg')
