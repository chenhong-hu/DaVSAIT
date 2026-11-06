"""Exercise CUDA and the project's generator/discriminator backward paths."""
import torch
import torchvision
from utils.config import get_cfg
from solver import VSAITSolver

print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, flush=True)
print('CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0), flush=True)
print('Architectures:', torch.cuda.get_arch_list(), flush=True)
x = torch.randn(1, 3, 32, 32, device='cuda')
conv = torch.nn.Conv2d(3, 8, 3).cuda()
conv(x).square().mean().backward()
torch.cuda.synchronize()
print('CUDA convolution backward OK', flush=True)
cfg = get_cfg()
cfg.merge_from_file('configs/vsait.yaml')
cfg.TASK_MODEL.IMAGE_LOGGING_FREQ = 0
solver = VSAITSolver(cfg).cuda()
batch = {k: {'image': torch.rand(1, 3, 256, 256, device='cuda') * 2 - 1}
         for k in ['source', 'target']}
for idx in [0, 1]:
    solver.zero_grad(set_to_none=True)
    outputs, targets = solver.forward(batch, optimizer_idx=idx)
    loss = solver.criterion(outputs, targets)
    assert torch.isfinite(loss).all(), 'Nonfinite loss'
    loss.backward()
    torch.cuda.synchronize()
    print('Optimizer {} forward/backward OK; loss={}'.format(idx, loss.item()), flush=True)
print('GPU verification passed', flush=True)
