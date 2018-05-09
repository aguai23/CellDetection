import cv2
from tf_unet import unet
import numpy as np
from matplotlib import pyplot as plt
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
data = cv2.imread("/data/Cell/data_svm/test/test/2018_01_15_09_11_11_30754_18024.jpg")
plt.imshow(data)
plt.show()
# average = np.average(data, axis=(0,1))
# print(average)
# std = np.std(data, axis=(0,1))
# print(std)
# data = (data - average) / std
# print(data)
data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
prediction = net.predict("/data/Cell/unet/model/model.ckpt25", data)
result = np.argmax(prediction, axis=3)
result = np.reshape(result, (result.shape[1], result.shape[2]))
plt.imshow(result)
plt.show()
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()