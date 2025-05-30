import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
import matplotlib.pyplot as plt


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        import math
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif beta_schedule == "pow":
        betas = np.linspace(0, 1, num_diffusion_timesteps)
        betas = pow(betas, 3)
        betas = betas * (beta_end - beta_start) + beta_start
    elif beta_schedule == "pow4":
        betas = np.linspace(0, 1, num_diffusion_timesteps)
        betas = pow(betas, 4)
        betas = betas * (beta_end - beta_start) + beta_start
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        # dataset, test_dataset = get_dataset(args, config)
        # train_loader = data.DataLoader(
        #     dataset,
        #     batch_size=config.training.batch_size,
        #     shuffle=True,
        #     num_workers=config.data.num_workers,
        # )

        train_loaders = []
        from datasets.seis_mat import SeisMatTrainDataset_clean, SeisMatValidationDataset_clean
        train_set = SeisMatTrainDataset_clean(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
                                              patch_size=self.config.data.image_size, pin_memory=True)  # f35_s256_o128
        dataloader = DataLoader(train_set, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
        train_loaders.append(dataloader)

        # from datasets.prepare_data.mat.bia2small_mat import generate_patch_from_mat,generate_patch_from_mat_v2
        # train_im_list = generate_patch_from_mat_v2(self.args.siesmic_dir, self.args.patch_size, stride=args.stride).astype(np.float32)
        # from datasets.DenoisingDatasets_seismic import SeisNpyTrainDataset
        # train_set = SeisNpyTrainDataset(im_list=train_im_list, length=self.args.cropped_data_num,
        #                                 pch_size=self.args.cropped_patch_size)
        # dataloader = DataLoader(train_set, batch_size=self.config.training.batch_size, shuffle=True,
        #                         num_workers=0, drop_last=True)
        # train_loaders.append(dataloader)


        # /home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128  D:\\datasets\\seismic\\marmousi\\f35_s256_o128
        # test_loaders = []
        # test_set = SeisMatValidationDataset_clean(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
        #                                           patch_size=self.config.data.image_size, pin_memory=True)
        # test_loader = DataLoader(test_set, batch_size=self.config.training.batch_size, shuffle=False,
        #                          num_workers=0, drop_last=True)
        # test_loaders.append(test_loader)
        # test_iter = iter(test_loader)

        print('dataloader done!')






        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            # for i, (x, y) in enumerate(train_loader):
            for i, x in enumerate(train_loaders[0]):
                n = x['H'].size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x['H'].to(self.device) #mcj
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)
                if step % 100 == 0:
                    logging.info(
                        f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        stochastic_variations = torch.zeros(config.sampling.num_samples,
                                            1, config.data.image_size,
                                            config.data.image_size)

        seed = 2025 #2025 2024
        torch.manual_seed(seed)

        x = torch.randn(
            config.sampling.num_samples,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            xs, x = self.sample_image(x, model, last=False)
            # x = self.sample_image(x, model, last=True)

        x = [inverse_data_transform(config, y) for y in x[-1]]

        for i in range(len(x)):
                 # tvu.save_image(
                    #     x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                    # )

                    # plt.gcf().set_size_inches(3, 3)
                    # # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
                    # plt.imshow(x[i][j].numpy().squeeze(), cmap=plt.cm.seismic, vmin=-1,
                    #            vmax=1)
                    # # plt.colorbar()  # 添加色标
                    # plt.axis('off')  # 关闭坐标轴
                    # plt.savefig(os.path.join(self.args.image_folder, f"{j}_{i}.png"), dpi=300,
                    #             bbox_inches='tight')
                stochastic_variations[i, :, :, :] = x[i]


        stochastic_variations_x_t = torch.zeros(config.sampling.num_samples * len(xs),
                                                    1, config.data.image_size,
                                                    config.data.image_size)
        stochastic_variations_x_t_rev = torch.zeros(config.sampling.num_samples * len(xs),
                                                1, config.data.image_size,
                                                config.data.image_size)
        for i in range(len(xs)):
            for j in range(xs[i].size(0)):
                stochastic_variations_x_t[j*len(xs)+i : j * len(xs)+(i+1), :, :, :] = inverse_data_transform(config, xs[i][j,:,:,:])

        for i in range(len(xs)):
            for j in range(xs[i].size(0)):
                # stochastic_variations_x_t_rev[j * len(xs) + (len(xs) - i-2): j * len(xs) + (len(xs) - i-1), :, :,:] = inverse_data_transform(
                #         config, xs[i][j, :, :, :])
                stochastic_variations_x_t_rev[j * len(xs) + i: j * len(xs) + (i + 1), :, :, :] = inverse_data_transform(
                    config, xs[len(xs)-i-1][j, :, :, :])

            # ######### plot stochastic_variations ###############
        from torchvision.utils import make_grid
        image_grid = make_grid(stochastic_variations, nrow=int(np.sqrt(config.sampling.num_samples)),
                               padding=8)  # args.num_samples int(np.sqrt(config.sampling.num_samples))
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1,
                   vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'image_grid_{}_gener_20250522.png'.format(
            config.sampling.ckpt_id)), dpi=300, bbox_inches='tight')
        # ######### plot stochastic_variations ###############
        from torchvision.utils import make_grid
        image_grid = make_grid(stochastic_variations_x_t, nrow=len(xs),
                               padding=8)  # args.num_samples
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(20, 20)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1,
                   vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_{}_x_t_20250522.png'.format(
            config.sampling.ckpt_id)), dpi=300, bbox_inches='tight')

        # ######### plot stochastic_variations ###############
        from torchvision.utils import make_grid
        image_grid = make_grid(stochastic_variations_x_t_rev, nrow=len(xs),
                               padding=8)  # args.num_samples
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(20, 20)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1,
                   vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_{}_x_t_rev_20250522.png'.format(
            config.sampling.ckpt_id)), dpi=300, bbox_inches='tight')


    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            if z1.size(0)>1:
                theta=[]
                result=[]
                for i in range(z1.size(0)):
                    theta.append(torch.acos(torch.sum(z1[i:i+1,:,:,:] * z2[i:i+1,:,:,:]) / (torch.norm(z1[i:i+1,:,:,:]) * torch.norm(z2[i:i+1,:,:,:]))))
                for i in range(z1.size(0)):
                    result.append(torch.sin((1 - alpha) * theta[i]) / torch.sin(theta[i]) * z1[i:i + 1, :, :, :]
                                  + torch.sin(alpha * theta[i]) / torch.sin(theta[i]) * z2[i:i + 1, :, :, :])

                return torch.cat(result, dim=0)
            else:
                theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
                return (
                    torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                    + torch.sin(alpha * theta) / torch.sin(theta) * z2
                )

        if config.sampling.num_samples > 1:
            z1=[]
            z2=[]

            for i in range(config.sampling.num_samples):
                z=torch.randn(
                1,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
                )
                z1.append(z)
            for i in range(config.sampling.num_samples):
                z=torch.randn(
                1,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
                )
                z2.append(z)
            z1=torch.cat(z1, dim=0)
            z2 = torch.cat(z2, dim=0)

        else:
            z1 = torch.randn(
                1,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            z2 = torch.randn(
                1,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))

        stochastic_variations_int = torch.zeros(x.size(0),
                                                1, config.data.image_size,
                                                config.data.image_size)
        # for i in range(x.size(0)):
        #     tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

        # for i in range(x.size(0)):
        #     stochastic_variations_int[i,:,:,:]=x[i,:,:,:]
        len_row= x.size(0)//config.sampling.num_samples
        for i in range(len_row):
            for j in range(config.sampling.num_samples):
                stochastic_variations_int[j*len_row+i,:,:,:]=x[j*len_row+i,:,:,:]

        from torchvision.utils import make_grid
        image_grid = make_grid(stochastic_variations_int, nrow=len_row,
                               padding=8)  # args.num_samples
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1,
                   vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_{}_int_20250522.png'.format(
            config.sampling.ckpt_id)), dpi=300, bbox_inches='tight')


    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
