import math
import torch
from torch.optim.optimizer import Optimizer
import itertools as it


class Ranger(Optimizer):
	def __init__(self, params, lr=1e-3, alpha=0.5, k=6, n_sma_threshhold=5, betas=(.95, 0.999),
				 eps=1e-5, weight_decay=0, amsgrad=True, transformer='softplus', smooth=50,
				 grad_transformer='square'):
		# parameter checks
		if not 0.0 <= alpha <= 1.0:
			raise ValueError(f'Invalid slow update rate: {alpha}')
		if not 1 <= k:
			raise ValueError(f'Invalid lookahead steps: {k}')
		if not lr > 0:
			raise ValueError(f'Invalid Learning Rate: {lr}')
		if not eps > 0:
			raise ValueError(f'Invalid eps: {eps}')

		# parameter comments:
		# beta1 (momentum) of .95 seems to work better than .90...
		# N_sma_threshold of 5 seems better in testing than 4.
		# In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

		# prep defaults and init torch.optim base
		defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
						n_sma_threshhold=n_sma_threshhold, eps=eps, weight_decay=weight_decay,
						smooth=smooth, transformer=transformer, grad_transformer=grad_transformer,
						amsgrad=amsgrad)
		super().__init__(params, defaults)

		# adjustable threshold
		self.n_sma_threshhold = n_sma_threshhold

		# look ahead params
		self.alpha = alpha
		self.k = k

		# radam buffer for state
		self.radam_buffer = [[None, None, None] for ind in range(10)]

		# self.first_run_check=0

		# lookahead weights
		# 9/2/19 - lookahead param tensors have been moved to state storage.
		# This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

		# self.slow_weights = [[p.clone().detach() for p in group['params']]
		#                     for group in self.param_groups]

		# don't use grad for lookahead weights
		# for w in it.chain(*self.slow_weights):
		#    w.requires_grad = False

	def __setstate__(self, state):
		print("set state called")
		super(Ranger, self).__setstate__(state)

	def step(self, closure=None):
		loss = None
		# note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
		# Uncomment if you need to use the actual closure...

		# if closure is not None:
		# loss = closure()

		# Evaluate averages and grad, update param tensors
		for group in self.param_groups:

			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data.float()
				if grad.is_sparse:
					raise RuntimeError('Ranger optimizer does not support sparse gradients')

				amsgrad = group['amsgrad']
				smooth = group['smooth']
				grad_transformer = group['grad_transformer']

				p_data_fp32 = p.data.float()

				state = self.state[p]  # get state dict for this param

				if len(state) == 0:  # if first time to run...init dictionary with our desired entries
					# if self.first_run_check==0:
					# self.first_run_check=1
					# print("Initializing slow buffer...should not see this at load from saved model!")
					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(p_data_fp32)
					state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
					if amsgrad:
						# Maintains max of all exp. moving avg. of sq. grad. values
						state['max_exp_avg_sq'] = torch.zeros_like(p.data)

						# look ahead weight storage now in state dict
					state['slow_buffer'] = torch.empty_like(p.data)
					state['slow_buffer'].copy_(p.data)

				else:
					state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
					state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

				# begin computations
				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']
				if amsgrad:
					max_exp_avg_sq = state['max_exp_avg_sq']

					# compute variance mov avg
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
				# compute mean moving avg
				exp_avg.mul_(beta1).add_(1 - beta1, grad)

				##transformer
				if grad_transformer == 'square':
					grad_tmp = grad ** 2
				elif grad_transformer == 'abs':
					grad_tmp = grad.abs()

				exp_avg_sq.mul_(beta2).add_((1 - beta2) * grad_tmp)

				if amsgrad:
					# Maintains the maximum of all 2nd moment running avg. till now
					torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
					# Use the max. for normalizing running avg. of gradient
					denomc = max_exp_avg_sq.clone()
				else:
					denomc = exp_avg_sq.clone()

				if grad_transformer == 'square':
					# pdb.set_trace()
					denomc.sqrt_()

				state['step'] += 1

				if group['weight_decay'] != 0:
					p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']
				step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

				# ...let's use calibrated alr
				if group['transformer'] == 'softplus':
					sp = torch.nn.Softplus(smooth)
					denomf = sp(denomc)
					p_data_fp32.addcdiv_(-step_size, exp_avg, denomf)

				else:

					denom = exp_avg_sq.sqrt().add_(group['eps'])
					p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

				p.data.copy_(p_data_fp32)

				# integrated look ahead...
				# we do it at the param level instead of group level
				if state['step'] % group['k'] == 0:
					slow_p = state['slow_buffer']  # get access to slow param tensor
					slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
					p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

		return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
