import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_tensorboard_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

from src.data.chexpert_datamodule import CheXpertDataModule
from src.models.cfg_diffusion import CustomClassConditionedUnet
from src.utils.helpers import _extract_into_tensor
from src.utils.constants import CHEXPERT_CLASSES
from src.models.conditional_ddpm_pipeline import ConditionalDDPMPipeline
from src.models.ambient_diffusion import make_ambient_batch, ambient_loss, AmbientDDPMPipeline

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert DDPM Training Script")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the CheXpert dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-chexpert-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument(
        "--num_epochs",
        type=int, 
        default=100)
    
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=25,
        help="How often to save images during training.",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=10,
        help="How often to save the model during training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay magnitude for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="The power value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay magnitude for EMA.",
    )

    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "neptune"],
        help="Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) for experiment tracking",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    # Additional CheXpert-specific arguments
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Run in debug mode with a small subset of data",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pin_memory for faster data transfer to GPU",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization"
    )

    parser.add_argument(
        "--original_dememorization",
        action="store_true",
        help="Use original dememorization method",
    )

    parser.add_argument(
        "--tau", type=float, default=3.0, help="Tau value for dememorization"
    )

    parser.add_argument(
        "--ambient",
        action="store_true",
        help="Use ambient diffusion model",
    )

    parser.add_argument(
        "--ambient_p",
        type=float,
        default=0.9,
        help="Bernoulli keep-prob for primary inpainting mask A",
    )

    parser.add_argument(
        "--ambient_delta",
        type=float,
        default=0.05,
        help="Extra drop-prob δ for secondary mask ˜A",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=7200)
    )  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if not is_tensorboard_available():
        raise ImportError(
            "Make sure to install tensorboard if you want to use it for logging during training."
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(
                        os.path.join(output_dir, "custom_unet_ema")
                    )

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "custom_unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "custom_unet_ema"),
                    CustomClassConditionedUnet,
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = CustomClassConditionedUnet.from_pretrained(
                    input_dir, subfolder="custom_unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = CustomClassConditionedUnet(
        sample_size=args.resolution,
    )

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=CustomClassConditionedUnet,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys()
    )
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the CheXpert DataModule
    datamodule = CheXpertDataModule(
        data_dir=args.data_dir,
        img_size=args.resolution,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        seed=args.seed,
        debug_mode=args.debug_mode,
        pin_memory=args.pin_memory,
    )

    # Prepare the data
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Get the dataloaders
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(datamodule.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Skip steps until resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["image"].to(weight_dtype)
            class_labels = batch["labels"].to(weight_dtype)

            guidance_dropout_prob = 0.1
            batch_size = clean_images.size(0)

            use_conditioning_mask = (
                torch.rand(batch_size, device=clean_images.device)
                >= guidance_dropout_prob
            )

            conditional_input = class_labels.clone()
            conditional_input[~use_conditioning_mask] = torch.zeros_like(
                class_labels[~use_conditioning_mask]
            )

            tau = getattr(args, "tau", 1.0)

            noise = torch.randn(
                clean_images.shape, dtype=weight_dtype, device=clean_images.device
            )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
            ).long()

            if args.ambient:
                noisy_images, A_mask = make_ambient_batch(
                    clean_images, noise_scheduler, timesteps
                )
            else:
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                if args.original_dememorization:
                    # Conditional prediction
                    cond_output = model(
                        sample=noisy_images,
                        timestep=timesteps,
                        class_labels=conditional_input,
                    ).sample

                    # Unconditional prediction
                    uncond_labels = torch.zeros_like(class_labels)
                    uncond_output = model(
                        sample=noisy_images,
                        timestep=timesteps,
                        class_labels=uncond_labels,
                    ).sample

                    # Magnitude difference calculation
                    diff = cond_output - uncond_output  # [B, C, H, W]
                    flat = diff.flatten(start_dim=1)  # [B, C*H*W]
                    magnitude_diff = torch.linalg.norm(flat, dim=1)

                    valid_mask = magnitude_diff <= tau
                    percent_filtered = (1.0 - valid_mask.float().mean()) * 100

                    if valid_mask.any():
                        if args.prediction_type == "epsilon":
                            loss = F.mse_loss(
                                cond_output[valid_mask].float(),
                                noise[valid_mask].float(),
                            )
                        elif args.prediction_type == "sample":
                            alpha_t = _extract_into_tensor(
                                noise_scheduler.alphas_cumprod,
                                timesteps,
                                (batch_size, 1, 1, 1),
                            )
                            snr_weights = alpha_t / (1 - alpha_t)
                            loss = snr_weights[valid_mask] * F.mse_loss(
                                cond_output[valid_mask].float(),
                                clean_images[valid_mask].float(),
                                reduction="none",
                            )
                            loss = loss.mean()
                        else:
                            raise ValueError(
                                f"Unsupported prediction type: {args.prediction_type}"
                            )
                    else:
                        loss = torch.tensor(
                            0.0, device=clean_images.device, requires_grad=True
                        )

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:
                    # Predict the noise residual
                    model_output = model(
                        sample=noisy_images,
                        timestep=timesteps,
                        class_labels=conditional_input,
                    ).sample

                    if args.ambient:
                        # Force sampling for ambient diffusion
                        args.prediction_type = "sample"

                    if args.prediction_type == "epsilon":
                        loss = F.mse_loss(
                            model_output.float(), noise.float()
                        )  # this could have different weights!

                    elif args.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod,
                            timesteps,
                            (clean_images.shape[0], 1, 1, 1),
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        if args.ambient:
                            loss = ambient_loss(
                                model_output,
                                clean_images,
                                A_mask,
                            )
                        else:
                            loss = snr_weights * F.mse_loss(
                                model_output.float(),
                                clean_images.float(),
                                reduction="none",
                            )
                        loss = loss.mean()
                    else:
                        raise ValueError(
                            f"Unsupported prediction type: {args.prediction_type}"
                        )

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    and global_step % args.checkpointing_steps == 0
                ):
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [
                                d
                                for d in os.listdir(args.output_dir)
                                if d.startswith("checkpoint")
                            ],
                            key=lambda x: int(x.split("-")[1]),
                        )

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for chkpt in checkpoints[
                                : len(checkpoints) - args.checkpoints_total_limit + 1
                            ]:
                                shutil.rmtree(os.path.join(args.output_dir, chkpt))

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if 'percent_filtered' in locals():
                logs["percent_filtered"] = percent_filtered.item()
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            accelerator.log(logs, step=global_step)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (epoch % args.save_images_epochs == 0) or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                if args.ambient:
                    pipeline = AmbientDDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                    p_mask=0.9,                 # keep the same masking probability you used in training
                )
                else:
                    pipeline = ConditionalDDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)

                # Create labels for generation
                class_labels = torch.zeros((14, 14), device=pipeline.device)
                # (args.eval_batch_size, 14), device=pipeline.device
                # CheXpert class labels
                CHEXPERT_CLASSES = [
                    "No Finding",
                    "Enlarged Cardiomediastinum",
                    "Cardiomegaly",
                    "Lung Opacity",
                    "Lung Lesion",
                    "Edema",
                    "Consolidation",
                    "Pneumonia",
                    "Atelectasis",
                    "Pneumothorax",
                    "Pleural Effusion",
                    "Pleural Other",
                    "Fracture",
                    "Support Devices",
                ]

                # Set the first few classes for visualization
                for i in range(len(CHEXPERT_CLASSES)):
                    class_labels[i, i % len(CHEXPERT_CLASSES)] = 1.0

                # Generate images with conditioning - only difference is passing class_labels
                if args.ambient:
                    result = pipeline(
                    generator=generator,
                    batch_size=len(CHEXPERT_CLASSES),
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="np",
                    )
                else:
                    result = pipeline(
                        generator=generator,
                        batch_size=len(CHEXPERT_CLASSES),  # args.eval_batch_size,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        output_type="np",
                        class_labels=class_labels,
                        guidance_scale=3.0,
                    )
                images = result["images"]

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # Keep the exact same image processing that works in the original code
                # Since our images are grayscale, we need to repeat across channels for visualization
                images = np.repeat(
                    images, 3, axis=3
                )  # Repeat the single channel three times to make RGB

                # denormalize the images and save to tensorboard (same as original)
                images_processed = (images * 255).round().astype("uint8")

                # Log images to tensorboard (same as original)
                if is_accelerate_version(">=", "0.17.0.dev0"):
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                else:
                    tracker = accelerator.get_tracker("tensorboard")

                tracker.add_images(
                    "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
                )

                # Also save as individual images for inspection
                os.makedirs(
                    os.path.join(args.output_dir, f"samples_epoch_{epoch}"),
                    exist_ok=True,
                )
                for i, image in enumerate(images_processed):
                    # Convert to PIL and save
                    Image.fromarray(image).save(
                        os.path.join(
                            args.output_dir,
                            f"samples_epoch_{epoch}",
                            f"sample_{CHEXPERT_CLASSES[i]}.png",
                        )
                    )

                # Save the model if it's a save_model_epoch
                if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    # save the model
                    pipeline.save_pretrained(
                        os.path.join(args.output_dir, f"model_epoch_{epoch}")
                    )

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# Run using: python conditional_ddpm_train_v2.py   --data_dir CheXpert-v1.0-small   --output_dir ./chest_xray_diffusion   --resolution 128   --train_batch_size 4  --num_epochs 100   --dataloader_num_workers 4   --learning_rate 1e-4   --use_ema   --mixed_precision fp16 --debug_mode
