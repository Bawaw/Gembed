#!/usr/bin/env python3

import sys

import torch
import lightning as pl
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects
from torch_scatter import scatter_mean
from transform import SubsetSample


# def frechet_mean(
#     model,
#     dataset,
#     n_input_samples,
#     init_template,
#     f_metric=lambda x, y: (x - y).pow(2).sum(-1),
#     n_iters=100,
#     device="cpu",
# ):
#     pl.seed_everything(42, workers=True)

#     # data transform
#     T = tgt.Compose([tgt.SamplePoints(n_input_samples), tgt.Center()])

#     # setup model
#     model = model.to(device)

#     # Compute Conditions
#     Zs = torch.concat(
#         [model.inverse(T(d.clone()).pos.to(device), None) for d in dataset], 0
#     )
#     Z_mean = model.inverse(T(init_template.clone()).pos.to(device), None)

#     # Compute point distribution for template
#     if init_template is not None:
#         z_template = model.pdm_inverse(
#             x=init_template.pos.clone().to(device), condition=Z_mean
#         )
#     else:
#         z_template = (torch.randn(10000, 3).to(device),)

#     # find the frechet mean
#     with torch.set_grad_enabled(True):
#         Z_mean = Z_mean.requires_grad_(True)

#         optimiser = torch.optim.Adam([Z_mean])

#         for i in range(n_iters):
#             optimiser.zero_grad()

#             distance = f_metric(Zs, Z_mean)
#             average_distance = distance.mean()

#             average_distance.backward()
#             optimiser.step()

#     # Synthesise the mean shape
#     X_mean = init_template.clone()
#     X_mean.pos = model.pdm_forward(
#         z=z_template,
#         condition=Z_mean,
#         return_time_steps=False,
#     )

#     X_mean = X_mean.cpu()

#     # plot the mean shape
#     plot_objects((X_mean, X_mean.pos[:, 0]))


from gembed.stats.geodesic import discrete_geodesic
from gembed.stats.frechet_mean import frechet_mean

def construct_template(
    model,
    dataset,
    init_template,
    local_metric_type,
    n_iters=100,
    device="cpu",
    snapshot_root=None,
    riemannian_pullback_kwargs={
        "optim_n_random_point_samples":10,
        "n_geodesic_cps":5,
    }
):
    pl.seed_everything(42, workers=True)

    f_sample_points = (
        tgt.SamplePoints(8192) if hasattr(dataset[0], "face") else SubsetSample(8192)
    )

    # data transform
    T = tgt.Compose(
        [
            f_sample_points,
            tgt.NormalizeScale(),
        ]
    )

    T_norm = tgt.NormalizeScale()

    # Setup Template
    X_mean = init_template.clone()
    X_mean_scalars = X_mean.pos[:, 0]

    # move data to device
    model = model.to(device)
    X_mean = X_mean.to(device)

    Z_mean = model.inverse(T(X_mean.clone()).pos, None,  apply_stn=True)
    z_mean = model.pdm_inverse(
        x=T_norm(X_mean.clone()).pos,
        # we sample the template for the condition
        condition=Z_mean,
        apply_stn=True,
    )

    # compute embeddings
    Zs = torch.concat(
        [model.inverse(T(d.clone()).pos.to(device), None, apply_stn=True) for d in dataset], 0
    )

    # frechet means
    if local_metric_type == "shape_mse":
        Xs = torch.stack([
            model.pdm.forward(z=z_mean, condition=Z[None]) for Z in Zs
        ])

        # plot_objects((Xs[0].cpu(), None),(Xs[1].cpu(), None),(Xs[2].cpu(), None), cmap="cool")

        f_global_metric = lambda x, y: 0.5*(x - y).pow(2).sum(-1).mean(-1)

        X_mean.pos = frechet_mean(
            Xs,
            init_mean=T_norm(X_mean.clone()).pos,
            f_global_metric=f_global_metric,
            verbose=False,
        )

        X_mean_2 = X_mean.clone()
        X_mean_2.pos = Xs.mean(0)

    elif local_metric_type == "latent_mse":
        Z_mean = frechet_mean(
            Zs,
            init_mean=Z_mean,
        )
        X_mean.pos = model.pdm.forward(z=z_mean, condition=Z_mean)

        # X_mean_2 = X_mean.clone()
        # X_mean_2.pos = model.pdm.forward(z=z_mean, condition=Zs.mean(0, keepdim=True))

    elif local_metric_type == "latent_hyperbolic":
        def poincare_distance(Zi, Zj):
            zi_norm = (Zi ** 2).sum(-1)
            zj_norm = (Zj ** 2).sum(-1)

            pq = (Zi - Zj).pow(2).sum(-1)
            dist = torch.arccosh(1 + 2 * pq / ((1 - zi_norm) * (1 - zj_norm)))
            return dist

        f_global_metric = lambda Z0, Z1: discrete_geodesic(Z0, Z1, f_local_metric=poincare_distance, n_iters=int(1e4), return_energy=True)[1]
        Z_mean = frechet_mean(
            Zs,
            init_mean=Z_mean,
            f_global_metric=f_global_metric,
            verbose=True
        )
        X_mean.pos = model.pdm.forward(z=z_mean, condition=Z_mean)

    # 5) interpolation in shape space (geodesic metric)
    if local_metric_type == "pullback_riemannian":
        z_template_optim = torch.randn(riemannian_pullback_kwargs["optim_n_random_point_samples"], 3).to(device)
        n_geodesic_cps = riemannian_pullback_kwargs["n_geodesic_cps"]
        z_template_optim_batched = z_template_optim.repeat(n_geodesic_cps-1,1)
        batch = torch.concat([i*torch.ones(z_template_optim.shape[0]) for i in range(n_geodesic_cps-1)]).long().to(z_template_optim.device)

        f_riemannian_metric = lambda Zi, Zj: scatter_mean((model.pdm.forward(z=z_template_optim_batched, condition=Zi, batch=batch) - model.pdm.forward(z=z_template_optim_batched, condition=Zj, batch=batch)).pow(2).sum(-1), batch)
        f_global_metric = lambda Z0, Z1: discrete_geodesic(Z0, Z1, f_local_metric=f_riemannian_metric, n_iters=1, return_energy=True)[1]
        Z_mean = frechet_mean(
            Zs,
            init_mean=Z_mean,
            f_global_metric=f_global_metric,
            verbose=True
        )
        X_mean.pos = model.pdm.forward(z=z_mean, condition=Z_mean)

    # move data objects back to cpu
    X_mean = X_mean.cpu()

    # plot and save results
    objects_and_scalars = [
        (T_norm(d), None) for d in dataset
    ]
    objects_and_scalars.append((X_mean.cpu(), X_mean.pos[:,0]))
    #objects_and_scalars.append((X_mean_2.cpu(), X_mean.pos[:,0]))

    if snapshot_root is not None:
        for i, sample_and_scalar in enumerate(objects_and_scalars):
            os.makedirs(snapshot_root, exist_ok=True)
            save_file_path = os.path.join(snapshot_root, f"interpolated_shape_t_{i/(len(objects_and_scalars) - 1)}.png")

            plot_objects(sample_and_scalar, snapshot_file_name=save_file_name, cmap="cool")

    else:
        plot_objects(*objects_and_scalars, cmap="cool")


# def construct_template(
#     model, dataset, init_template, n_samples=10, n_iters=5, device="cpu"
# ):
#     pl.seed_everything(42, workers=True)

#     template = tgt.Center()(init_template.clone())
#     old_template = template.clone()

#     n_samples = min(n_samples, len(dataset))

#     if hasattr(template, "face"):
#         T_template = tgt.Compose(
#             [
#                 # tgt.SamplePoints(4096),
#             ]
#         )
#     else:
#         T_template = tgt.Compose(
#             [
#                 # SubsetSample(4096),
#             ]
#         )
#     if hasattr(dataset[0], "face"):
#         T = tgt.Compose(
#             [
#                 # tgt.SamplePoints(4096),
#             ]
#         )
#     else:
#         T = tgt.Compose(
#             [
#                 # SubsetSample(4096),
#             ]
#         )

#     model = model.to(device)
#     transformed_template = T_template(template.clone()).to(device)
#     template = template.to(device)

#     for e in range(n_iters):

#         # point embedding for template mesh
#         template_z = model.pdm_inverse(
#             x=template.pos, condition=model.inverse(transformed_template.pos, None)
#         )

#         reconstructions = []

#         for i in range(n_samples):
#             # grab data sample
#             X_data = tgt.Center()(dataset[i].clone())
#             X_data_transformed = T(X_data.clone()).to(device)
#             X_data = X_data.to(device)

#             # reconstruct data sample in template format
#             X_rec = model.pdm_forward(
#                 z=template_z,
#                 condition=model.inverse(X_data_transformed.pos, None),
#                 return_time_steps=False,
#             )

#             reconstructions.append(X_rec.cpu())

#         template.pos = torch.stack(reconstructions).mean(0).to(device)

#     template = template.cpu()
#     plot_objects((old_template, old_template.pos[:, 0]), (template, template.pos[:, 0]))

#     return template
