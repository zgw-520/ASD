markdown
复制代码
# 面部表情识别项目

## 项目简介
本项目实现了一个面部表情识别系统，利用卷积神经网络（CNN）对输入的图像进行表情分类。该系统能够实时识别七种不同的面部表情：愤怒、厌恶、恐惧、快乐、悲伤、惊讶和中立。通过使用摄像头捕捉实时视频流，系统将自动检测人脸并进行表情预测。

## 主要特性
- 实时人脸检测和表情识别
- 支持七种面部表情分类
- 使用深度学习模型进行高效的推理
- 基于 OpenCV 的图像处理和视频捕捉

## 环境要求
- Python 3.x
- PyTorch
- torchvision
- OpenCV

## 安装指南
1. 克隆该仓库：
   ```bash
   git clone https://github.com/zgw-520/Facial_expression_recognition.git
   cd Facial_expression_recognition
安装依赖：
bash
复制代码
pip install -r requirements.txt
使用方法
确保你的摄像头已连接。
运行 main.py 文件：
bash
复制代码
python main.py
程序运行后，将打开一个窗口，显示实时的视频流和检测到的表情标签。
按 q 键退出程序。
项目结构
bash
复制代码
data/
    fer2013.csv                      # 数据集
models/
    facial_expression_model.pth       # 训练好的模型
    haarcascade_frontalface_default.xml  # Haar Cascade 分类器文件
__init__.py                          # 包的初始化文件
main.py                               # 主程序
predict.py                            # 预测脚本
训练模型（可选）
如果你希望自己训练模型，可以使用以下步骤：

准备数据集并将其放置在 data/fer2013.csv 中。
编写数据加载和模型训练代码（请参考相关 PyTorch 文档）。
保存训练好的模型到 models/facial_expression_model.pth。
示例结果
可以通过运行程序观察实时的人脸表情识别结果。以下是一个示例截图（如果有的话，可以在此添加截图）。

贡献
欢迎任何形式的贡献！如果你发现任何问题或有改进建议，请创建一个 issue 或提交 pull request。

联系信息
如有问题或建议，请联系：

姓名：zgw-520
邮箱：540634587@qq.com
GitHub: zgw-520
