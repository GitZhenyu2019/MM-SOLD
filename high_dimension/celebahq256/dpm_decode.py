"""
Canonical DiffAE decode path: DPM-Solver++(2M) on a uniform log-SNR grid.
"""
import numpy as np
import torch
import tensorrt as trt

import config as cfg

__all__ = ["make_renderer"]


def make_renderer(engine_path=None, nsteps=None, batch=None, dev="cuda",
                  first_order=False):
    """
    Return (render, ctx) where render(x_T, z) -> images in [0, 1].
    """
    engine_path = cfg.DIFFAE_TRT_ENGINE if engine_path is None else engine_path
    nsteps = cfg.DECODE_NSTEPS if nsteps is None else nsteps
    batch = cfg.DECODE_BATCH if batch is None else batch

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
        open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    xb = torch.empty(batch, 3, 256, 256, device=dev)
    tb = torch.empty(batch, device=dev)
    zb = torch.empty(batch, 512, device=dev)
    eps = torch.empty(batch, 3, 256, 256, device=dev)
    bufs = {"x": xb, "t": tb, "cond": zb, "eps": eps}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        ctx.set_tensor_address(nm, bufs[nm].data_ptr())

    betas = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
    abar = np.cumprod(1.0 - betas)
    lam_all = np.log(np.sqrt(abar) / np.sqrt(1.0 - abar))     # half log-SNR
    lam_t = np.linspace(lam_all[999], lam_all[0], nsteps)     # ascending lambda
    ts = np.interp(lam_t, lam_all[::-1], np.arange(1000)[::-1])   # descending t
    assert ts[0] > ts[-1]
    sg_t = 1.0 / np.sqrt(1.0 + np.exp(2.0 * lam_t))
    al_t = np.exp(lam_t) * sg_t
    al = torch.tensor(al_t, dtype=torch.float32, device=dev)
    sg = torch.tensor(sg_t, dtype=torch.float32, device=dev)
    lam = torch.tensor(lam_t, dtype=torch.float64, device=dev)
    tvals = torch.tensor(ts, dtype=torch.float32, device=dev)

    def render(x_T, z):
        x = x_T.clone()
        zb.copy_(z)
        x0_prev, h_prev = None, None
        N = len(ts)
        for k in range(N):
            xb.copy_(x)
            tb.fill_(float(tvals[k]))
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            x0 = ((x - sg[k] * eps) / al[k]).clamp(-1, 1)      # data prediction
            if k == N - 1:                                     # land clean
                x = x0
                break
            h = float(lam[k + 1] - lam[k])
            if x0_prev is None or first_order:
                D = x0
            else:                                              # 2nd-order multistep
                r = h_prev / h
                D = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev
            x = (sg[k + 1] / sg[k]) * x - al[k + 1] * (np.exp(-h) - 1.0) * D
            x0_prev, h_prev = x0, h
        return (x + 1) / 2

    return render, ctx
