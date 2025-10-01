# Acoustic FWI with illumination compensation

# * This code block is modified from 
# https://github.com/ar4/deepwave/blob/master/docs/example_custom_imaging_condition.py

# * The idea is to put a loop over time for the C++/CUDA forward modeling solver. 

# * Originally the C++/CUDA forward solver calcualte all the time steps from $i_t=1,...,n_t$. Now by adding a time loop in Python, each call of the C++/CUDA forward solver only solve for `step_ratio` time steps. Here `step_ratio` is a number. In this example, it is 4.

# * The source illumination compensation is calculated by integrating the square of forward wavefield over shots and time steps. 



from typing import Optional, Union, List, Tuple
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import deepwave
from deepwave.common import setup_propagator, diff, create_or_pad, zero_interior, downsample_and_movedim




# Method 2 is broken into parts. First we do some setup of the propagator and then
# call the Autograd Function that is defined further down
def method2_scalar(v: Tensor,
                   grid_spacing: Union[int, float, List[float], Tensor],
                   dt: float,
                   source_amplitudes: Optional[Tensor] = None,
                   source_locations: Optional[Tensor] = None,
                   receiver_locations: Optional[Tensor] = None,
                   accuracy: int = 4,
                   pml_width: Union[int, List[int]] = 20,
                   pml_freq: Optional[float] = None,
                   max_vel: Optional[float] = None,
                   survey_pad: Optional[Union[int,
                                              List[Optional[int]]]] = None,
                   wavefield_0: Optional[Tensor] = None,
                   wavefield_m1: Optional[Tensor] = None,
                   psiy_m1: Optional[Tensor] = None,
                   psix_m1: Optional[Tensor] = None,
                   zetay_m1: Optional[Tensor] = None,
                   zetax_m1: Optional[Tensor] = None,
                   origin: Optional[List[int]] = None,
                   nt: Optional[int] = None,
                   model_gradient_sampling_interval: int = 1,
                   freq_taper_frac: float = 0.0,
                   time_pad_frac: float = 0.0):

    (models, source_amplitudes_l, wavefields,
     pml_profiles, sources_i_l, receivers_i_l,
     dy, dx, dt, nt, n_shots,
     step_ratio, model_gradient_sampling_interval,
     accuracy, pml_width_list) = \
        setup_propagator([v], 'scalar', grid_spacing, dt,
                         [wavefield_0, wavefield_m1, psiy_m1, psix_m1,
                          zetay_m1, zetax_m1],
                         [source_amplitudes],
                         [source_locations], [receiver_locations],
                         accuracy, pml_width, pml_freq, max_vel,
                         survey_pad,
                         origin, nt, model_gradient_sampling_interval,
                         freq_taper_frac, time_pad_frac)
    v = models[0]
    wfc, wfp, psiy, psix, zetay, zetax = wavefields
    source_amplitudes = source_amplitudes_l[0]

    sources_i = sources_i_l[0]
    receivers_i = receivers_i_l[0]
    ay, ax, by, bx = pml_profiles
    dbydy = diff(by, accuracy, dy)
    dbxdx = diff(bx, accuracy, dx)

    (wfc, wfp, psiy, psix, zetay, zetax, diag_Hessian, receiver_amplitudes) = \
        method2_func(
            v, source_amplitudes, wfc, wfp, 
            psiy, psix, zetay, zetax, 
            ay, ax, by, bx, dbydy, dbxdx, 
            sources_i, receivers_i, dy, dx, 
            dt, nt, step_ratio * model_gradient_sampling_interval, 
            accuracy, pml_width_list, n_shots, receiver_locations
        )

    receiver_amplitudes = downsample_and_movedim(receiver_amplitudes,
                                                 step_ratio, freq_taper_frac,
                                                 time_pad_frac)

    return wfc, wfp, psiy, psix, zetay, zetax, diag_Hessian, receiver_amplitudes


# Using an Autograd Function enables us to define our own forward and backward
class Method2ForwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, source_amplitudes, wfc, wfp, psiy, psix, zetay, zetax,
                ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i, dy, dx,
                dt, nt, step_ratio, accuracy, pml_width, n_shots, receiver_locations):

        # Ensure that the Tensors are contiguous in memory as the C/CUDA code
        # assumes that they are
        v = v.contiguous()
        source_amplitudes = source_amplitudes.contiguous()

        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()

        # Create the wavefields, or add padding (for the finite difference
        # stencil) if they already exist
        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape)
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype,
                              size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype,
                              size_with_batch)
        # Zero the interior of the PML-related wavefields
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)

        # Define some needed values
        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots

        # Allocate some temporary and output Tensors. dwdv stores snapshots
        # from forward propagation for use during backpropagation.
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        dwdv = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        # Set the coordinates of the edges of where the PML calculations
        # need to be performed.
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        # Define quantities that depend on whether we are running on a CPU or GPU
        if v.is_cuda:
            aux = v.get_device()  # the CUDA device number
            if v.requires_grad:
                dwdv.resize_(nt // step_ratio, n_shots, *v.shape)
                dwdv.fill_(0)
            if receivers_i is not None:
                receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            # Select the CUDA function that will be used to propagate
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_iso_6_float_forward
                else:
                    forward = deepwave.dll_cuda.scalar_iso_8_float_forward
            else:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_iso_2_double_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_iso_4_double_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_iso_6_double_forward
                else:
                    forward = deepwave.dll_cuda.scalar_iso_8_double_forward
        else:  # Running on CPU
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad:
                dwdv.resize_(n_shots, nt // step_ratio, *v.shape)
                dwdv.fill_(0)
            if receivers_i is not None:
                receiver_amplitudes.resize_(n_shots, nt, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_iso_6_float_forward
                else:
                    forward = deepwave.dll_cpu.scalar_iso_8_float_forward
            else:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_iso_2_double_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_iso_4_double_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_iso_6_double_forward
                else:
                    forward = deepwave.dll_cpu.scalar_iso_8_double_forward

        # accumulate receiver amplitudes
        receiver_amplitudes_accum = torch.zeros_like(receiver_amplitudes).contiguous()
        # accumulate dwdv
        dwdv_accum = torch.zeros_like(dwdv).contiguous()
        # print("dwdv_accum.shape", dwdv_accum.shape)


        slice_4Hessian = (slice(None),
                          slice(fd_pad+pml_width[0], -fd_pad-pml_width[1]), 
                          slice(fd_pad+pml_width[2], -fd_pad-pml_width[3]))
        

        diag_Hessian = torch.zeros_like(v[slice_4Hessian[1:]]).contiguous()


        # print("diag_Hessian.shape", diag_Hessian.shape)




        for i in range(0, nt // step_ratio):

            chunk = source_amplitudes[i * step_ratio:(i + 1) * step_ratio, :, :].contiguous()

            forward(v.data_ptr(), chunk.data_ptr(), wfc.data_ptr(),
                    wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
                    psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
                    zetax.data_ptr(), dwdv.data_ptr(),
                    receiver_amplitudes.data_ptr(), ay.data_ptr(),
                    ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                    dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
                    receivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                    1 / dx**2, dt**2, step_ratio, n_shots, ny, nx, n_sources_per_shot,
                    n_receivers_per_shot, step_ratio, v.requires_grad, pml_y0,
                    pml_y1, pml_x0, pml_x1, aux)

         
            # accumulate receiver amplitudes (shot gathers) and dwdv (derivative of wavefile w.r.t velocity)
            # reciever amplitude is used as fowarded data
            # dwdv is used for backpropagation
            if v.is_cuda:
                receiver_amplitudes_accum[i * step_ratio: (i + 1) * step_ratio, :, :] = receiver_amplitudes[0: step_ratio, :, :]
                # print("v.is_cude. TRUE. dwdv_accum.shape", dwdv_accum.shape)
                dwdv_accum[i, :, :, :] = dwdv[0, :, :, :]
            else:
                receiver_amplitudes_accum[:, i * step_ratio: (i + 1) * step_ratio, :] = receiver_amplitudes[:, 0: step_ratio, :]
                # print("v.is_cude. FALSE. dwdv_accum.shape", dwdv_accum.shape)
                dwdv_accum[:, i, :, :] = dwdv[:, 0, :, :]

            if step_ratio % 2 != 0:
                wfc, wfp = wfp, wfc
                psiy, psiyn = psiyn, psiy
                psix, psixn = psixn, psix

            # if i % 200 == 0:
            #     print("v.shape", v.shape, "diag_Hessian.shape", diag_Hessian.shape)
            #     print("wfp.shape", wfp.shape)
            #     print("pml_width", pml_width)
            #     print("wfp[slice_4Hessian].shape", wfp[slice_4Hessian].shape)

            # Use wfp to calculate the illumination compensation            
            #R.-E. Plessix and W. A. Mulder, 2004, Type 1
            # diag_Hessian += (wfp[slice_4Hessian]**2 ).sum(dim=0)
            #R.-E. Plessix and W. A. Mulder, 2004, Type 2
            # diag_Hessian += (wfp[slice_4Hessian]**4 ).sum(dim=0)
            #R.-E. Plessix and W. A. Mulder, 2004, Type 3
            diag_Hessian += (wfp[slice_4Hessian]**2 ).sum(dim=0)
        



        # receiver illumination using Plessix and Mulder, 2004, Type 3
        recv_ymin = (receiver_locations[0, 0, 0] + 1) * dy    # minimum x coordinate of the receiver
        recv_ymax = (receiver_locations[0, -1, 0] + 1) * dy   # maximum x coordinate of the receiver
        # print("recv_ymin", recv_ymin)
        # print("recv_ymax", recv_ymax)
        # receiver illumination
        # recv_illum = torch.zeros_like(diag_Hessian)
        nyy = diag_Hessian.shape[0]
        nxx = diag_Hessian.shape[1]
        x_coords = (torch.arange(1, nxx+1) * dx).repeat(nyy, 1).to(device)
        y_coords = (torch.arange(1, nyy+1) * dy).repeat(nxx, 1).T.to(device)

        # print("x_coords.shape", x_coords.shape)
        # print("y_coords.shape", y_coords.shape)
        # print("x_coords", x_coords)
        # print("y_coords", y_coords)

        recv_illum = torch.asinh( (recv_ymax - y_coords)/x_coords ) - torch.asinh( (recv_ymin - y_coords)/x_coords )



        # for iy in range(diag_Hessian.shape[0]): # z coordinates
        #     for ix in range(diag_Hessian.shape[1]):  # x coordinates
        #         x_cord = (ix+1)*dx  # z coordinates
        #         y_cord = (iy+1)*dy  # x coordinates

        #         recv_illum[iy, ix] = torch.asinh( (recv_ymax - y_cord)/x_cord ) - torch.asinh( (recv_ymin - y_cord)/x_cord )

        diag_Hessian = diag_Hessian * recv_illum





        # THIS IS THE ORIGINAL VERSION OF THE CODE
        # Call the C/CUDA function to propagate forward
        # if wfc.numel() > 0 and nt > 0:
        #     forward(v.data_ptr(), source_amplitudes.data_ptr(), wfc.data_ptr(),
        #             wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
        #             psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
        #             zetax.data_ptr(), dwdv.data_ptr(),
        #             receiver_amplitudes.data_ptr(), ay.data_ptr(),
        #             ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
        #             dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
        #             receivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
        #             1 / dx**2, dt**2, nt, n_shots, ny, nx, n_sources_per_shot,
        #             n_receivers_per_shot, step_ratio, v.requires_grad, pml_y0,
        #             pml_y1, pml_x0, pml_x1, aux)


        # Save data needed for backpropagation
        if (v.requires_grad or source_amplitudes.requires_grad
                or wfc.requires_grad or wfp.requires_grad or psiy.requires_grad
                or psix.requires_grad or zetay.requires_grad
                or zetax.requires_grad):
            # dwdv is replaced by dwdv_accum, where all the derivatives of wavefile w.r.t velocity are accumulated.
            ctx.save_for_backward(v, ay, ax, by, bx, dbydy, dbxdx, sources_i,
                                receivers_i, dwdv_accum)
            
            ctx.dy = dy
            ctx.dx = dx
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = source_amplitudes.requires_grad

        # Remove the padding added for the finite difference stencil
        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        # sH = (slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        # print(s, sH)
        if nt % 2 == 0:
            return (wfc[s], wfp[s], psiy[s], psix[s], zetay[s], zetax[s], diag_Hessian, 
                    receiver_amplitudes_accum)
        # wfc: wavefield current, wfp: wavefield previous
        else:
            return (wfp[s], wfc[s], psiyn[s], psixn[s], zetay[s], zetax[s], diag_Hessian,
                    receiver_amplitudes_accum)






    @staticmethod
    @once_differentiable
    def backward(ctx, wfc, wfp, psiy, psix, zetay, zetax, diag_Hessian, grad_r):
        (v, ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i,
         dwdv) = ctx.saved_tensors


        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        fd_pad = accuracy // 2


        v = v.contiguous()
        grad_r = grad_r.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        dwdv = dwdv.contiguous()


        size_with_batch = (n_shots, *v.shape)
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype,
                              size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype,
                              size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype,
                               size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype,
                               size_with_batch)
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)

        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        zetayn = torch.zeros_like(zetay)
        zetaxn = torch.zeros_like(zetax)
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        if v.requires_grad:
            grad_v.resize_(*v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 3 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 3 * fd_pad)
        pml_x0 = min(pml_width[2] + 3 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 3 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad and n_shots > 1:
                grad_v_tmp.resize_(n_shots, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(nt, n_shots, n_sources_per_shot)
                grad_f.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_iso_6_float_backward
                else:
                    backward = deepwave.dll_cuda.scalar_iso_8_float_backward
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_iso_2_double_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_iso_4_double_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_iso_6_double_backward
                else:
                    backward = deepwave.dll_cuda.scalar_iso_8_double_backward
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad and aux > 1 and deepwave.use_openmp:
                grad_v_tmp.resize_(aux, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(n_shots, nt, n_sources_per_shot)
                grad_f.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_iso_6_float_backward
                else:
                    backward = deepwave.dll_cpu.scalar_iso_8_float_backward
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_iso_2_double_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_iso_4_double_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_iso_6_double_backward
                else:
                    backward = deepwave.dll_cpu.scalar_iso_8_double_backward

        v2dt2 = v**2 * dt**2
        wfp = -wfp

        if wfc.numel() > 0 and nt > 0:
            backward(v2dt2.data_ptr(), grad_r.data_ptr(), wfc.data_ptr(),
                     wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
                     psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
                     zetax.data_ptr(), zetayn.data_ptr(), zetaxn.data_ptr(),
                     dwdv.data_ptr(), grad_f.data_ptr(),
                     grad_v.data_ptr(), grad_v_tmp_ptr, ay.data_ptr(),
                     ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                     dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
                     receivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                     1 / dx**2, nt, n_shots, ny, nx,
                     n_sources_per_shot * source_amplitudes_requires_grad,
                     n_receivers_per_shot, step_ratio, v.requires_grad, pml_y0,
                     pml_y1, pml_x0, pml_x1, aux)
            
        # print("grad_v.shape", grad_v.shape)
        # print("grad_f.shape", grad_f.shape)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return grad_v, grad_f, wfc[s], -wfp[s], psiy[s], psix[
                s], zetay[s], zetax[s], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        else:
            return grad_v, grad_f, wfp[s], -wfc[s], psiyn[s], psixn[
                s], zetayn[s], zetaxn[s], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None




def method2_func(*args):

    return Method2ForwardFunc.apply(*args)


def method_2():
    loss_fn = torch.nn.MSELoss()
    v = v_init.clone().requires_grad_()

    diag_Hessian, receiver_amplitudes = method2_scalar(v,
                                         dx,
                                         dt,
                                         source_amplitudes=source_amplitudes,
                                         source_locations=source_locations,
                                         receiver_locations=receiver_locations,
                                         max_vel=max_vel)[-2:]
    loss = 1e6 * loss_fn(receiver_amplitudes, d_true)
    loss.backward()

    return v.grad.detach(), diag_Hessian.detach()