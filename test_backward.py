import torch
import numpy as np
from diff_skewed_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

# Dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ajustes mínimos para rasterización
settings = GaussianRasterizationSettings(
    image_height=64,
    image_width=64,
    tanfovx=np.tan(np.radians(60/2)),
    tanfovy=np.tan(np.radians(60/2)),
    bg=torch.tensor([0.0, 0.0, 0.0], device=device),
    scale_modifier=1.0,
    viewmatrix=torch.eye(4, dtype=torch.float32, device=device),
    projmatrix=torch.eye(4, dtype=torch.float32, device=device),
    sh_degree=0,
    campos=torch.tensor([0.0, 0.0, -2.0], device=device),
    prefiltered=False,
    debug=True  # importante: activa debug si tu código CUDA imprime condicionalmente
)
rasterizer = GaussianRasterizer(settings).to(device)

# Definir una única Gaussiana con requires_grad
means3D = torch.tensor([[0.0, 0.0, 5.0]], device=device, requires_grad=True)
scales   = torch.tensor([[1.0, 1.0, 1.0]], device=device, requires_grad=True)
rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, requires_grad=True)  # sin rotación
skews    = torch.tensor([[0.0, 0.0, 0.0]], device=device, requires_grad=True)
skew_sensitivity = torch.tensor([1000.0], device=device, requires_grad=True)

colors_precomp = torch.tensor([[1.0, 0.0, 0.0]], device=device, requires_grad=True)  # Rojo
opacities     = torch.tensor([[1.0]], device=device, requires_grad=True)

# Forward: rasterizar
rendered, _ = rasterizer(
    means3D=means3D,
    means2D=None,
    opacities=opacities,
    colors_precomp=colors_precomp,
    scales=scales,
    rotations=rotations,
    skews=skews,
    skew_sensitivity=skew_sensitivity
)

# Definir una pérdida simple y backward
loss = rendered.sum()
loss.backward()

# Forzar sincronización para asegurarte de que todos los printf de CUDA aparezcan
torch.cuda.synchronize()

# Imprimir gradientes
print("Gradiente de means3D:", means3D.grad)
print("Gradiente de scales:",   scales.grad)
print("Gradiente de skews:",    skews.grad)
