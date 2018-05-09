from tf_unet import image_util, unet
output_path = "/data/Cell/unet/model/"
data_provider = image_util.ImageDataProvider("/data/Cell/unet/*.jpg")

net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
trainer = unet.Trainer(net, optimizer="adam")
path = trainer.train(data_provider, output_path, training_iters=32, epochs=30)