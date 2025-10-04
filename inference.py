from twin_gpt.models.lit_gpt import LitGPT

# load the pth for LitGPT
model = LitGPT.load_from_checkpoint("logs/checkpoints/twin_gpt-epoch=00-val_loss=nan.ckpt")
print (model)