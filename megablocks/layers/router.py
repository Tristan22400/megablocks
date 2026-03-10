from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch


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
