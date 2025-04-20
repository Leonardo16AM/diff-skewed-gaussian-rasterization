#!/usr/bin/env python
# coding: utf‑8
"""Gaussian Viewer – 4 gaussianas (RGBY) con control completo de la gaussiana 0 y rotación automática."""

import sys, math, numpy as np, torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QGridLayout,
                             QCheckBox)
from PyQt5.QtGui import QImage, QPixmap
from diff_skewed_gaussian_rasterization import (
    GaussianRasterizer, GaussianRasterizationSettings
)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def look_at(eye, center, up=np.array([0,1,0], dtype=np.float32)):
    f = (center - eye)
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

def rotation_matrix_from_euler(x, y, z):
    """Crea una matriz de rotación a partir de ángulos de Euler (en radianes)."""
    # Rotación en X
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ], dtype=torch.float32)
    
    # Rotación en Y
    Ry = torch.tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ], dtype=torch.float32)
    
    # Rotación en Z
    Rz = torch.tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Orden de aplicación: primero Z, luego Y, finalmente X
    R = Rx @ Ry @ Rz
    return R


class GaussianViewer(QMainWindow):
    # ────────────────────────── INIT ────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Cámara
        self.cam_yaw = self.cam_pitch = 0.0
        self.cam_pos    = torch.tensor([0., 0., -2.], device=self.device)
        self.cam_target = torch.tensor([0., 0.,  5.], device=self.device)
        
        # Parámetros para la rotación automática
        self.auto_rotation = True
        self.rotation_speed = 0.01
        self.rotation_radius = 4.0
        self.rotation_height = 0.5
        self.rotation_angle = 0.0
        self.center_point = torch.tensor([0., 0., 5.], device=self.device)  # Centro de las gaussianas

        # 4 gaussianas – posiciones, escalas, skew
        self.gaussian_pos = torch.tensor(
            [[-1., -1., 5.],
             [ 1., -1., 5.],
             [ 1.,  1., 5.],
             [-1.,  1., 5.]], device=self.device)                          # (4,3)

        self.gaussian_scale = torch.full((4, 3), 0.25, device=self.device)   # (4,3)
        self.gaussian_skew  = torch.zeros((4, 3), device=self.device)        # (4,3)
        self.skew_sensitivity = torch.full((4,), 1000.0, device=self.device) # (4,)

        self.colors = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0]],
                                   dtype=torch.float32, device=self.device)
        self.opacities = torch.ones((4,1), device=self.device)
        
        # Inicializar rotaciones (quaterniones para 3D GS)
        self.rotations = torch.zeros((4,4), device=self.device)
        self.rotations[:,3] = 1.0  # Componente W del quaternion = 1 (sin rotación)
        
        # Ángulos de Euler para la gaussiana 0 (en radianes)
        self.euler_angles = torch.zeros(3, device=self.device)

        # UI y render loop
        self.last_mouse_pos = None; self.mouse_pressed = False
        self.init_ui(); self.setup_renderer()

        self.timer = QTimer(self); self.timer.timeout.connect(self.update_render)
        self.timer.start(16)  # ≈60 fps

    # ────────────────────────── UI ──────────────────────────────────────────
    def init_ui(self):
        self.setWindowTitle("Gaussian Viewer")
        self.setMinimumSize(960, 680)

        main_w = QWidget(); main_l = QHBoxLayout(main_w)
        # --- Ventana de render
        self.render_view = QLabel(alignment=Qt.AlignCenter)
        self.render_view.setMinimumSize(512, 512)
        self.render_view.setMouseTracking(True)
        self.render_view.mousePressEvent   = self.mouse_press_event
        self.render_view.mouseMoveEvent    = self.mouse_move_event
        self.render_view.mouseReleaseEvent = self.mouse_release_event
        self.render_view.setFocusPolicy(Qt.StrongFocus)
        self.render_view.keyPressEvent     = self.key_press_event
        main_l.addWidget(self.render_view, 3)

        # --- Panel de controles
        ctl_p = QWidget(); ctl_l = QVBoxLayout(ctl_p)

        # 0) Controles de rotación automática
        rot_g = QGroupBox("Rotación Automática"); rot_l = QVBoxLayout()
        
        # Checkbox para activar/desactivar rotación
        self.auto_rot_cb = QCheckBox("Habilitar rotación")
        self.auto_rot_cb.setChecked(self.auto_rotation)
        self.auto_rot_cb.stateChanged.connect(self.toggle_auto_rotation)
        rot_l.addWidget(self.auto_rot_cb)
        
        # Slider para velocidad de rotación
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Velocidad:"))
        self.speed_slider = QSlider(Qt.Horizontal, minimum=1, maximum=50, value=int(self.rotation_speed*1000))
        self.speed_slider.valueChanged.connect(self.update_rotation_params)
        self.speed_label = QLabel(f"{self.rotation_speed:.3f}")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        rot_l.addLayout(speed_layout)
        
        # Slider para radio de rotación
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radio:"))
        self.radius_slider = QSlider(Qt.Horizontal, minimum=1, maximum=100, value=int(self.rotation_radius*10))
        self.radius_slider.valueChanged.connect(self.update_rotation_params)
        self.radius_label = QLabel(f"{self.rotation_radius:.1f}")
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_label)
        rot_l.addLayout(radius_layout)
        
        # Slider para altura de rotación
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Altura:"))
        self.height_slider = QSlider(Qt.Horizontal, minimum=-50, maximum=50, value=int(self.rotation_height*10))
        self.height_slider.valueChanged.connect(self.update_rotation_params)
        self.height_label = QLabel(f"{self.rotation_height:.1f}")
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        rot_l.addLayout(height_layout)
        
        rot_g.setLayout(rot_l)
        ctl_l.addWidget(rot_g)

        # 1) Posición
        self.pos_sl, self.pos_lb = self._add_vec3_sliders(
            parent_layout=ctl_l,
            title="Posición g0",
            init=self.gaussian_pos[0], rng=(-5.0, 5.0), step=0.1)

        # 2) Escala
        self.scale_sl, self.scale_lb = self._add_vec3_sliders(
            parent_layout=ctl_l,
            title="Scale g0",
            init=self.gaussian_scale[0], rng=(0.01, 1.0), step=0.01)

        # 3) Skew
        self.skew_sl, self.skew_lb = self._add_vec3_sliders(
            parent_layout=ctl_l,
            title="Skew g0",
            init=self.gaussian_skew[0], rng=(-1.0, 1.0), step=0.01)

        # 4) Rotación (sliders para los ángulos de Euler)
        self.rot_sl, self.rot_lb = self._add_vec3_sliders(
            parent_layout=ctl_l,
            title="Rotación g0 (XYZ)",
            init=self.euler_angles, rng=(-2*math.pi, 2*math.pi), step=0.01)

        # 5) Skew sensitivity (scalar)
        sens_g = QGroupBox("Skew Sensitivity g0"); sens_l = QHBoxLayout()
        self.sens_slider = QSlider(Qt.Horizontal, minimum=0, maximum=5000,
                                   value=int(self.skew_sensitivity[0].item()))
        self.sens_slider.valueChanged.connect(self.update_gaussian0)
        self.sens_label = QLabel(f"{self.skew_sensitivity[0]:.0f}")
        sens_l.addWidget(self.sens_slider); sens_l.addWidget(self.sens_label)
        sens_g.setLayout(sens_l); ctl_l.addWidget(sens_g)

        # Info cámara
        self.camera_info = QLabel(); ctl_l.addWidget(self.camera_info)
        ctl_l.addStretch(); main_l.addWidget(ctl_p, 1)

        self.setCentralWidget(main_w); self.setFocusPolicy(Qt.StrongFocus)

    # Helper para crear sliders XYZ
    def _add_vec3_sliders(self, parent_layout, title, init, rng, step):
        grp = QGroupBox(title); grid = QGridLayout()
        sliders, labels = [], []
        lo, hi = rng
        scale = 1 / step
        for i, ax in enumerate("XYZ"):
            grid.addWidget(QLabel(f"{ax}:"), i, 0)
            sl = QSlider(Qt.Horizontal, minimum=int(lo*scale), maximum=int(hi*scale),
                         value=int(init[i].item()*scale))
            sl.valueChanged.connect(self.update_gaussian0)
            lbl = QLabel(f"{init[i]:.2f}")
            grid.addWidget(sl, i, 1); grid.addWidget(lbl, i, 2)
            sliders.append(sl); labels.append(lbl)
        grp.setLayout(grid); parent_layout.addWidget(grp)
        return sliders, labels

    # ─────────────────────  RASTERIZER SET‑UP  ───────────────────────────────
    def setup_renderer(self):
        # Reducir el FOV para tener menos distorsión en perspectiva
        self.fov, self.aspect, self.near, self.far = 60., 1., 0.1, 100.
        self.view_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        self.update_rasterizer()

    def get_projection_matrix(self):
        # La función de Graph‑Deco recibe fovX y fovY *en radianes*
        proj = getProjectionMatrix(
            znear = self.near,
            zfar  = self.far,
            fovX  = math.radians(self.fov),   # fov horizontal
            fovY  = math.radians(self.fov))   # fov vertical
        return torch.tensor(proj, dtype=torch.float32, device=self.device).T

    def update_rasterizer(self):
        rs = GaussianRasterizationSettings(
            image_height=512, 
            image_width=512,
            tanfovx=np.tan(np.radians(self.fov/2)),
            tanfovy=np.tan(np.radians(self.fov/2)),
            bg=torch.tensor([0.05,0.05,0.05], device=self.device),
            scale_modifier=1.0,
            viewmatrix=self.view_matrix,  # Ya está transpuesta desde update_view_matrix
            projmatrix=self.get_projection_matrix(),
            sh_degree=0,
            campos=self.cam_pos,
            prefiltered=False, 
            debug=False)
        self.rasterizer = GaussianRasterizer(rs)

    # ─────────────  MATRIZ DE VISTA y PROYECCIÓN A PANTALLA  ────────────────
    def update_view_matrix(self):
        eye = self.cam_pos.cpu().numpy()
        center = self.cam_target.cpu().numpy()
        up = np.array([0,1,0], dtype=np.float32)
        view_matrix = look_at(eye, center, up)
        return torch.tensor(view_matrix, dtype=torch.float32, device=self.device)

    def project_to_screen(self, pts3D, H=512, W=512):
        N = pts3D.shape[0]
        pts4 = torch.cat([pts3D, torch.ones((N,1), device=self.device)], 1)
        clip = (self.get_projection_matrix() @ (self.view_matrix @ pts4.T)).T
        ndc = clip[:,:3] / clip[:,3:4]
        xy = torch.empty((N,2), device=self.device)
        xy[:,0] = ( ndc[:,0]*0.5 + 0.5) * (W-1)
        xy[:,1] = (-ndc[:,1]*0.5 + 0.5) * (H-1)
        return xy

    # ─────────────────────── UI CALLBACKS ───────────────────────────────────
    def update_gaussian0(self):
        # Pos
        for i,(sl,lbl) in enumerate(zip(self.pos_sl, self.pos_lb)):
            self.gaussian_pos[0,i] = sl.value()/10.0; lbl.setText(f"{self.gaussian_pos[0,i]:.1f}")
        # Scale
        for i,(sl,lbl) in enumerate(zip(self.scale_sl, self.scale_lb)):
            self.gaussian_scale[0,i] = sl.value()/100.0; lbl.setText(f"{self.gaussian_scale[0,i]:.2f}")
        # Skew
        for i,(sl,lbl) in enumerate(zip(self.skew_sl, self.skew_lb)):
            self.gaussian_skew[0,i] = sl.value()/100.0; lbl.setText(f"{self.gaussian_skew[0,i]:.2f}")
        # Rotación (ángulos de Euler)
        for i,(sl,lbl) in enumerate(zip(self.rot_sl, self.rot_lb)):
            self.euler_angles[i] = sl.value()/100.0; lbl.setText(f"{self.euler_angles[i]:.2f}")
            
        # Convertir ángulos de Euler a quaternion y matriz de rotación
        # Primero creamos la matriz de rotación
        R = rotation_matrix_from_euler(
            self.euler_angles[0],  # X
            self.euler_angles[1],  # Y
            self.euler_angles[2]   # Z
        ).to(self.device)
        
        rotated_skew = torch.matmul(R, self.gaussian_skew[0])
        self.gaussian_skew[0] = rotated_skew
        
        # Update skew labels
        for i, lbl in enumerate(self.skew_lb):
            lbl.setText(f"{self.gaussian_skew[0,i]:.2f}")
        
        # Actualizar el quaternion para g0 (conversión desde matriz de rotación a quaternion)
        tr = R[0,0] + R[1,1] + R[2,2]
        
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
            
        # Guardar el quaternion
        self.rotations[0,0] = qx
        self.rotations[0,1] = qy
        self.rotations[0,2] = qz
        self.rotations[0,3] = qw
        
        # Sensitivity
        self.skew_sensitivity[0] = float(self.sens_slider.value())
        self.sens_label.setText(f"{self.skew_sensitivity[0]:.0f}")
    
    def toggle_auto_rotation(self, state):
        self.auto_rotation = (state == Qt.Checked)
    
    def update_rotation_params(self):
        # Actualizar velocidad, radio y altura según los sliders
        self.rotation_speed = self.speed_slider.value() / 1000.0
        self.speed_label.setText(f"{self.rotation_speed:.3f}")
        
        self.rotation_radius = self.radius_slider.value() / 10.0
        self.radius_label.setText(f"{self.rotation_radius:.1f}")
        
        self.rotation_height = self.height_slider.value() / 10.0
        self.height_label.setText(f"{self.rotation_height:.1f}")

    def mouse_press_event(self, ev):
        if ev.button()==Qt.LeftButton:
            self.mouse_pressed=True; self.last_mouse_pos=ev.pos(); self.render_view.setFocus()
            # Desactivar rotación automática cuando el usuario manipula la cámara
            self.auto_rot_cb.setChecked(False)
            self.auto_rotation = False
    
    def mouse_release_event(self, ev):
        if ev.button()==Qt.LeftButton: self.mouse_pressed=False
    
    def mouse_move_event(self, ev):
        if not self.mouse_pressed: return
        dx,dy = ev.x()-self.last_mouse_pos.x(), ev.y()-self.last_mouse_pos.y()
        self.cam_yaw-=dx*0.005; self.cam_pitch-=dy*0.005
        self.cam_pitch = max(-math.pi/2+1e-3, min(math.pi/2-1e-3, self.cam_pitch))
        self.last_mouse_pos=ev.pos()

    def key_press_event(self, ev):
        # Desactivar rotación automática cuando el usuario manipula la cámara
        self.auto_rot_cb.setChecked(False)
        self.auto_rotation = False
        
        speed=0.2
        dir_f=torch.tensor([math.cos(self.cam_pitch)*math.sin(self.cam_yaw),
                            math.sin(self.cam_pitch),
                            math.cos(self.cam_pitch)*math.cos(self.cam_yaw)], device=self.device)
        right=torch.tensor([math.sin(self.cam_yaw-math.pi/2),0.,
                            math.cos(self.cam_yaw-math.pi/2)], device=self.device)
        up=torch.tensor([0.,1.,0.], device=self.device)
        k=ev.key()
        if   k==Qt.Key_W: self.cam_pos+=dir_f*speed
        elif k==Qt.Key_S: self.cam_pos-=dir_f*speed
        elif k==Qt.Key_A: self.cam_pos-=right*speed
        elif k==Qt.Key_D: self.cam_pos+=right*speed
        elif k==Qt.Key_Q: self.cam_pos+=up*speed
        elif k==Qt.Key_E: self.cam_pos-=up*speed
        # Tecla R para reiniciar la rotación automática
        elif k==Qt.Key_R: 
            self.auto_rot_cb.setChecked(True)
            self.auto_rotation = True

    # ───────────────────── ROTACIÓN AUTOMÁTICA ───────────────────────────────
    def update_auto_rotation(self):
        if not self.auto_rotation:
            return
            
        # Actualizar ángulo de rotación
        self.rotation_angle += self.rotation_speed
        if self.rotation_angle > 2 * math.pi:
            self.rotation_angle -= 2 * math.pi
            
        # Calcular nueva posición de cámara en un círculo alrededor del centro
        x = self.center_point[0] + self.rotation_radius * math.cos(self.rotation_angle)
        z = self.center_point[2] + self.rotation_radius * math.sin(self.rotation_angle)
        y = self.center_point[1] + self.rotation_height
        
        # Actualizar posición de cámara
        self.cam_pos = torch.tensor([x, y, z], device=self.device)
        
        # Apuntar al centro de las gaussianas
        direction = self.center_point - self.cam_pos
        direction_norm = direction / torch.norm(direction)
        
        # Calcular yaw y pitch a partir de la dirección
        self.cam_yaw = math.atan2(direction_norm[0].item(), direction_norm[2].item())
        self.cam_pitch = math.asin(direction_norm[1].item())

    # ───────────────────────── RENDER LOOP ───────────────────────────────────
    def update_render(self):
        # Actualizar rotación automática si está activada
        if self.auto_rotation:
            self.update_auto_rotation()
        
        # Recalcular matrices
        dir_f = torch.tensor([math.cos(self.cam_pitch)*math.sin(self.cam_yaw),
                            math.sin(self.cam_pitch),
                            math.cos(self.cam_pitch)*math.cos(self.cam_yaw)], device=self.device)
        self.cam_target = self.cam_pos + dir_f
        self.view_matrix = self.update_view_matrix()
        self.update_rasterizer()

        self.camera_info.setText(
            f"Cam: ({self.cam_pos[0]:.1f},{self.cam_pos[1]:.1f},{self.cam_pos[2]:.1f})")

        means2D=torch.zeros_like(self.gaussian_pos, dtype=self.gaussian_pos.dtype, device="cuda")
        gaussians_homogeneous = torch.cat([
            self.gaussian_pos, 
            torch.ones((4,1), device=self.device)
        ], dim=1)  # [4,4]
        gaussians_camera = (self.view_matrix @ gaussians_homogeneous.T).T  # [4,4]
        

        rendered,_=self.rasterizer(
            means3D=self.gaussian_pos, means2D=means2D,
            opacities=self.opacities, colors_precomp=self.colors,
            scales=self.gaussian_scale, rotations=self.rotations,
            skews=self.gaussian_skew, skew_sensitivity=self.skew_sensitivity)
        

        img=(rendered.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        h,w,_=img.shape
        self.render_view.setPixmap(QPixmap.fromImage(
            QImage(img.tobytes(),w,h,3*w,QImage.Format_RGB888)))


# ──────────────────────────────── MAIN ───────────────────────────────────────
if __name__=="__main__":
    app=QApplication(sys.argv)
    viewer=GaussianViewer(); viewer.show()
    sys.exit(app.exec_())