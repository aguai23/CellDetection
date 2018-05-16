import os
from model import unet
from util import util
from data.test_data_provider import ImageTestProvider
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


net = unet.Unet(layers=5, features_root=16, channels=3, n_class=2)
data_provider = ImageTestProvider("/data/Cell/data_svm/test", batch_size=2, data_suffix='.jpg', mask_suffix='_mask.jpg',
                                  is_shuffle=True)
batch_mean_dice = 0
for batch in data_provider:
    image, mask = batch
    prediction = net.predict("/data/Cell/unet/more_epoch/model.ckpt95", image)
    batch_mean_dice += util.mean_dice(prediction, mask)
mean_dice = batch_mean_dice / len(data_provider)
print(mean_dice)
