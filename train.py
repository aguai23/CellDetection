from util import image_util, trainer
from model.unet import Unet
output_path = "/data/Cell/unet/model3/"
data_provider = image_util.ImageDataProvider("/data/Cell/unet/*.jpg")

net = Unet(layers=3, features_root=32, channels=3, n_class=2)
trainer = trainer.Trainer(net, optimizer="adam")
path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)