import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2


# 定义面部表情类别
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# 定义卷积神经网络模型（与训练时的结构相同）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, len(classes))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(model_path, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(img, transform):
    img = transform(img)
    img = img.unsqueeze(0)  # 添加批次维度
    return img


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = 'models/facial_expression_model.pth'
    model = load_model(model_path, device)

    # 定义预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(),  # 确保是灰度图
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 手动指定 Haar Cascade 文件路径
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("Failed to load Haar Cascade classifier for face detection.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # 提取脸部区域
            face = gray[y:y + h, x:x + w]
            try:
                # 预处理图像
                img = preprocess_image(face, transform).to(device)

                # 预测
                with torch.no_grad():
                    output = model(img)
                    _, predicted = torch.max(output, 1)
                    emotion = classes[predicted.item()]

                # 绘制矩形和标签
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        # 显示结果
        cv2.imshow('Facial Expression Recognition', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
