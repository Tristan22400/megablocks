from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch
import torch.distributed as dist


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1,keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        if self.args.moe_expert_choice:
            # Get probability for each token
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k # Use top k as the capacity to match regular MoEs
            logits = self.layer(x)
            scores = logits.softmax(dim=-1) # [batch_size, seq_len, num_experts]
            expert_weights, expert_indices = torch.topk(scores.transpose(1,2), (capacity * sq) // self.args.moe_num_experts, dim=-1) # [batch_size, num_experts, k]
        elif self.args.moe_expert_choice_grouped:
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k # Use top k as the capacity to match regular MoEs
            logits = self.layer(x.view(-1, x.shape[-1])) # [bs & sq, num_experts]
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(scores.transpose(0,1),  (capacity * bs * sq) // self.args.moe_num_experts, dim=-1) # [num_experts, k]
        else:
            logits = self.layer(x.view(-1, x.shape[-1]))
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights, p=self.args.moe_normalize_expert_weights,dim=-1, keepdim=True)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, logits, expert_weights, expert_indices


class LossFreeRouter(torch.nn.Module):
    """Auxiliary-loss-free load balancing router.

    Implements the method from Wang et al. (2024), "Auxiliary-Loss-Free Load
    Balancing Strategy for Mixture-of-Experts". An expert-wise additive bias is
    applied to routing logits before top-k selection to steer tokens toward
    underused experts. Supports both softmax and sigmoid gating:
      - Softmax: scores = softmax(logits + bias)
      - Sigmoid: scores = sigmoid(logits + bias)
    Both use proportional bias update: b[i] += u * (c_avg - c[i]).

    Reference: https://arxiv.org/abs/2408.15664
    """

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # Gate type: "softmax" (experts compete) or "sigmoid" (independent gates).
        self.gate_type = getattr(args, 'moe_gate_type', 'softmax')
        assert self.gate_type in ('softmax', 'sigmoid'), (
            f"Unknown gate type: {self.gate_type!r}. Must be 'softmax' or 'sigmoid'.")

        # Learned routing projection (identical to LearnedRouter).
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

        # --- Auxiliary-loss-free balancing state ---
        # Additive bias applied to logits before top-k. Registered as a buffer
        # so it is saved in state_dict, moved with .to(device), but is NOT an
        # nn.Parameter (optimizer never touches it, autograd does not track it).
        self.register_buffer(
            'expert_bias',
            torch.zeros(args.moe_num_experts, device=args.device,
                        dtype=torch.float32))

        # Bias update rate u (Algorithm 1, Wang et al. 2024).
        # Paper optimal: u = 0.001.
        speed = getattr(args, 'moe_bias_update_speed', None)
        self.bias_update_speed = speed if speed is not None else 0.001

        # Profiling: set _profile_update_bias = True to measure update_bias timing.
        self._profile_update_bias = False
        self._last_update_bias_ms = 0.0

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        # Compute raw routing logits.
        logits = self.layer(x.view(-1, x.shape[-1]))  # [T, E]

        # Add the load-balancing bias. .detach() severs the bias from the
        # computation graph: during backward, d(loss)/d(expert_bias) = 0,
        # so the router weights W receive only the task loss gradient.
        biased_logits = logits + self.expert_bias.detach().to(logits.dtype)

        # Route using biased logits.
        if self.gate_type == 'sigmoid':
            scores = biased_logits.sigmoid()
        else:
            scores = biased_logits.softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1, keepdim=True)

        expert_indices = (
            _uniform_expert_assignment(
                expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )

        # Return UNBIASED logits for z-loss computation. The z-loss
        # regularizes the raw logit magnitudes, not the biased ones.
        return scores, logits, expert_weights, expert_indices

    @torch.no_grad()
    def update_bias(self, tokens_per_expert: torch.Tensor):
        """Update expert bias from batch load (Algorithm 1, Wang et al. 2024).

        Constant update: b[i] += u * sign(c_avg - c[i]).

        Args:
            tokens_per_expert: [E] tensor of token counts per expert.
        """
        # All-reduce token counts across distributed workers so that all ranks
        # compute identical bias updates and stay synchronized.
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM)

        total = tokens_per_expert.sum()
        if total == 0:
            return

        # Error on raw counts: e[i] = c_avg - c[i].
        c = tokens_per_expert.float()
        c_avg = total.float() / self.args.moe_num_experts
        error = c_avg - c

        # Constant update (Algorithm 1, step 4): b[i] += u * sign(e[i]).
        self.expert_bias.add_(error.sign(), alpha=self.bias_update_speed)

    @torch.no_grad()
    def update_bias_local(self, tokens_per_expert: torch.Tensor):
        """Update expert bias from *already-reduced* token counts.

        Same as update_bias but skips the all-reduce (caller is responsible
        for reducing token counts across ranks beforehand).
        """
        total = tokens_per_expert.sum()
        if total == 0:
            return
        c = tokens_per_expert.float()
        c_avg = total.float() / self.args.moe_num_experts
        error = c_avg - c
        self.expert_bias.add_(error.sign(), alpha=self.bias_update_speed)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Handle loading checkpoints from LearnedRouter (missing buffers)."""
        for buf_name in ('expert_bias',):
            key = prefix + buf_name
            if key not in state_dict:
                state_dict[key] = torch.zeros(
                    self.args.moe_num_experts,
                    device=self.args.device,
                    dtype=torch.float32)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)


class RandomRouter(torch.nn.Module):
    """Uniform random token-to-expert assignment.

    Bypasses the learned router entirely: each token is assigned to top-k
    experts sampled uniformly at random (without replacement). Expert weights
    are set to 1/k so that each selected expert contributes equally.

    This serves as an ablation baseline to isolate the contribution of learned
    routing to scaling dynamics and expert specialization.
    """

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # Keep the projection layer for state_dict compatibility when loading
        # a pretrained checkpoint for analysis, but it is never used in forward.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x):
        T = x.view(-1, x.shape[-1]).shape[0]
        E = self.args.moe_num_experts
        K = self.args.moe_top_k

        # Sample K experts per token uniformly without replacement.
        # argsort of random values is an efficient GPU-vectorized permutation.
        expert_indices = torch.argsort(
            torch.rand(T, E, device=x.device), dim=-1
        )[:, :K]  # [T, K]

        # Uniform weights: each selected expert contributes equally.
        expert_weights = torch.full(
            (T, K), 1.0 / K, device=x.device, dtype=x.dtype)

        # Dummy scores and logits for interface compatibility.
        # scores: uniform distribution (no routing signal).
        # logits: zeros (no z-loss contribution desired).
        scores = torch.full(
            (T, E), 1.0 / E, device=x.device, dtype=x.dtype)
        logits = torch.zeros(T, E, device=x.device, dtype=x.dtype)

        return scores, logits, expert_weights, expert_indices
