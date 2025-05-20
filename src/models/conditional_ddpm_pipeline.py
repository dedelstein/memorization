# Create a conditional pipeline that matches the original closely
from diffusers import DDPMPipeline
import torch


class ConditionalDDPMPipeline(DDPMPipeline):
    """DDPM Pipeline with class conditioning support"""

    def __call__(
        self,
        batch_size=1,
        generator=None,
        num_inference_steps=1000,
        output_type="pil",
        class_labels=None,
        guidance_scale=1.5,
        return_dict=True,
    ):
        # Initialize with random noise (exactly like original)
        image = torch.randn(
            (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            ),
            generator=generator,
            device=self.device,
        )

        # Setup the scheduler (exactly like original)
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising process
        for t in self.scheduler.timesteps:
            # Only difference is we pass class_labels to the model
            with torch.no_grad():
                if guidance_scale > 1.0 and class_labels is not None:
                    # Conditional pass
                    cond_output = self.unet(
                        image, t, class_labels=class_labels
                    ).sample

                    # Unconditional pass
                    uncond_labels = torch.zeros_like(class_labels)
                    uncond_output = self.unet(
                        image, t, class_labels=uncond_labels
                    ).sample

                    # Combine with guidance scale
                    model_output = uncond_output + guidance_scale * (
                        cond_output - uncond_output
                    )
                else:
                    # Standard pass with conditioning
                    model_output = self.unet(
                        image, t, class_labels=class_labels
                    ).sample

            # Scheduler step (exactly like original)
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        # Final processing (exactly like original)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return dict(images=image)
