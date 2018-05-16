import cv2
from model import unet
import numpy as np
from matplotlib import pyplot as plt
net = unet.Unet(layers=3, features_root=32, channels=3, n_class=2)
data = cv2.imread("/data/Cell/cut_img/2018-03-01_14_41_56/2018-03-01_14_41_56_40960_23040.jpg")
print(data.shape)
plt.imshow(data[...,::-1])
plt.show()
# average = np.average(data, axis=(0,1))
# print(average)
# std = np.std(data, axis=(0,1))
# print(std)
# data = (data - average) / std
# print(data)
data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
prediction = net.predict("/data/Cell/unet/model3/model.ckpt95", data)
result = np.argmax(prediction, axis=3)
result = np.reshape(result, (result.shape[1], result.shape[2]))
print(result.shape)
plt.imshow(result, cmap="gray")
plt.show()
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()