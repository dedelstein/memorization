from src.data.chexpert_datamodule import CheXpertDataModule




# Example usage
data_module = CheXpertDataModule(
    data_dir="/home/prg/GitHub/memorization/CheXpert-v1.0-small",
    img_size=224,
    batch_size=16,
    num_workers=4,
    debug_mode=True,
)
data_module.prepare_data()
data_module.setup(stage="fit")
train_loader = data_module.train_dataloader()

for batch in train_loader:
    print(batch["image"].shape, batch["labels"].shape)
    break