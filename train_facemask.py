import tensorflow as tf
#  tf.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2

# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
# Xây dựng trình phân tích các cú pháp với đối số và phân tích các đối số



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")# Đường dẫn đến bộ dữ liệu đầu vào của khuôn mặt và mặt với mặt nạ
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")# Đường dẫn đến biểu đồ lịch sử đào tạo, sẽ được tạo bằng cách sử dụng matplotlib
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")# Đường dẫn đến mô hình phân loại khẩu trang
args = vars(ap.parse_args())

# các thông số train
INIT_LR = 1e-4  # khởi tạo tốc độ học tập ban đầu
EPOCHS = 20  # số lần mà thuật toán học tập sẽ hoạt động thông qua toàn bộ tập dữ liệu, chọn 20 để có độ chính xác cao
BS = 80  # Batch size (số lượng mẫu dữ liệu trong một lần huấn luyện)

# Lấy danh sách các hình ảnh trong dataset và khởi tạo
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
# lấy tất cả ảnh trong dataset
imagePaths = list(paths.list_images(args["dataset"]))
# Khởi tạo mảng data và labels(nhãn)
data = []
labels = []

# tiền xử lý ảnh
# tạo vòng lập để duyệt các ảnh trong dataset
for imagePath in imagePaths:
    # Trích xuất nhãn lớp từ tên tệp
    label = imagePath.split(os.path.sep)[-2]
    image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(224, 224))  # thay đổi kích thước thành 224 × 224 hoặc 229x229 pixel
    image = tf.keras.preprocessing.image.img_to_array(image)  # chuyển đổi sang định dạng mảng
    # chia tỷ lệ cường độ pixel trong hình ảnh đầu vào thành phạm vi [-1, 1] (thông qua hàm preprocess)
    # image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.keras.applications.densenet.preprocess_input(image)
    # Cập nhật danh sách dữ liệu và nhãn
    data.append(image)
    labels.append(label)
# Chuyển đổi dữ liệu và nhãn thành mảng numpy
data = np.array(data, dtype="float32")
labels = np.array(labels)
# thực hiện one-hot (đưa dữ liệu hạng mục về dạng số) trên nhãn
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)
# sau khi thực hiện ta được định dạng sau
# array([[1., 0.],
#        [1., 0.],
#        [1., 0.],
#        ...,
#        [0., 1.],
#        [0., 1.],
#        [0., 1.]], dtype=float32)
# (Pdb)
# mỗi phần tử của mảng nhãn của mình bao gồm một mảng trong đó chỉ có một chỉ mục là “hot” (e.i., 1)

#  Phân vùng dữ liệu vào đào tạo và kiểm tra chia tách bằng cách sử dụng 75% dữ liệu để đào tạo và 25% còn lại để kiểm tra bằng cách sử dụng scikit-learn’s
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)
# Xây dựng trình tạo hình ảnh đào tạo để tăng cường dữ liệu trong đó các tham số xoay, zoom, cắt, dịch chuyển và lật ngẫu nhiên
aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Thiết lập tinh chỉnh 3 bước:
# 1
# Tải mạng InceptionV3 và DensetNet121, đảm bảo các bộ lớp FC-fully connected đầu được bỏ lại
baseModel = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
#baseModel = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(299, 299, 3)))
# 2

# Đặt FC đầu lên trên mô hình cơ sở
headModel = baseModel.output
headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
headModel = tf.keras.layers.Dense(128, activation="relu")(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel) #bỏ qua một số các đơn vị trong mạng để tránh overfilting
headModel = tf.keras.layers.Dense(2, activation="softmax")(headModel)
# thay thế mô hình FC cho cái cũ
model = tf.keras.models.Model(inputs=baseModel.input, outputs=headModel)
# 3
# tạo vòng lập trên tất cả các lớp trong mô hình cơ sở và đóng băng chúng để chúng được cập nhật trong quá trình đào tạo đầu tiên
for layer in baseModel.layers:
    layer.trainable = False

# Bước huấn luyện
# Biên dịch mô hình với trình tối ưu hóa Adam
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# Huấn luyện phía đầu của mạng
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
#  đưa ra dự đoán trên tập kiểm tra
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# Đối với mỗi hình ảnh trong bộ thử nghiệm, cần tìm chỉ mục của nhãn với xác suất dự đoán lớn nhất tương ứng
predIdxs = np.argmax(predIdxs, axis=1)
# Hiển thị kết quả phân loại được định dạng
print(classification_report(testY.argmax(axis=1), predIdxs,  # type: ignore
	target_names=lb.classes_))
# tuần tự hóa mô hình thành đĩa
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5") 

# Chuyển đổi mô hình đào tạo
print("[INFO] covert model...")
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, './Web/static/models_desenet')

# vẽ sơ Training Loss and Accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])