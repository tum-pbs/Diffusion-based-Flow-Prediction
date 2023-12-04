#usr/bin/python3
import torch,math

kernel_dx = torch.Tensor( [[-1,  0,  1],
                            [-2,  0,  2],
                            [-1,  0,  1]] ) * (1/8)
kernel_dy = torch.Tensor( [[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]] ) * (1/8)
kernel_dx  = kernel_dx.view((1,1,3,3))
kernel_dy  = kernel_dy.view((1,1,3,3))

def rescale_unit(x,y):
    mag=torch.sqrt(x*x+y*y)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if mag[i][j].item()>1e-3:
                x[i][j] /= mag[i][j]
                y[i][j] /= mag[i][j]
    return x,y

def get_airfoil_unit_vector(airfoil_shape):
    shape=airfoil_shape.T
    shape=shape.view(1,1,shape.shape[0],shape.shape[1])
    dx_shape=torch.nn.functional.conv2d(shape,kernel_dx,padding=1)
    dy_shape=torch.nn.functional.conv2d(shape,kernel_dy,padding=1)
    dx_shape,dy_shape=rescale_unit(dx_shape[0][0],dy_shape[0][0])
    return dx_shape,dy_shape

def get_force_pressure(pressure_field,dx_shape,dy_shape,cell_length=2/128):
    pressure=pressure_field.T
    drag_x = torch.sum(dx_shape* pressure*cell_length) 
    drag_y = torch.sum(dy_shape* pressure*cell_length)
    return drag_x.item(),drag_y.item()

def get_force_vis(ux_field,uy_field,dx_shape,dy_shape,viscosity,cell_length=2/128):
    ux_field=ux_field.T
    uy_field=uy_field.T
    ux_field=ux_field.view(1,1,ux_field.shape[0],ux_field.shape[1])
    uy_field=uy_field.view(1,1,uy_field.shape[0],uy_field.shape[1])
    vorticity= torch.nn.functional.conv2d(uy_field, kernel_dx, padding=1)-torch.nn.functional.conv2d(ux_field, kernel_dy, padding=1)
    drag_x=torch.sum(dy_shape*vorticity*viscosity*cell_length)
    drag_y=torch.sum(-1*dx_shape*viscosity*cell_length)
    return drag_x.item(),drag_y.item()

def get_lift_drag_coef(airfoil_shape,pressure_field,ux_field,uy_field,AoA,velocity,viscosity=1e-5,density=1,cell_length=2/128):
    with torch.no_grad():
        dx_shape,dy_shape=get_airfoil_unit_vector(airfoil_shape)
        force_pressure_x,force_pressure_y=get_force_pressure(pressure_field=pressure_field,dx_shape=dx_shape,dy_shape=dy_shape,cell_length=cell_length)
        force_vis_x,force_vis_y=get_force_vis(ux_field=ux_field,uy_field=uy_field,dx_shape=dx_shape,dy_shape=dy_shape,viscosity=viscosity,cell_length=cell_length)
        Fx=force_pressure_x+force_vis_x
        Fy=force_pressure_y+force_vis_y
        angle=AoA/180*math.pi
        drag=Fy*math.sin(angle)+Fx*math.cos(angle)
        size=1
        energy=0.5*density*velocity*velocity*size
        return drag/energy
