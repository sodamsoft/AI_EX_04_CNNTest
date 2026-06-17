# AI_EX_04_CNNTest
CNN으로 mnist 데이터셋 구현

TensorFlow와 Keras를 활용하여 98.96%의 정확도로 MNIST 손글씨 숫자를 분류.
검증 데이터(Validation Data)를 추가하여 과적합을 모니터링하며 안정적인 학습 진행

# 모델 구조
 * Feature Extraction:
  - Conv2D (32 filters, 3x3, ReLU) + MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3, ReLU) + MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3, ReLU)
 * Classification:
  - Flatten
  - Dense (64 units, ReLU)
  - Dense (10 units, Softmax)

# 결과
  훈련 정확도(99.40%)와 검증 정확도(98.96%)의 차이가 매우 적어, 
  모델이 특정 데이터에 과적합(Overfitting)되지 않고 새로운 손글씨 이미지에 대해서도 높은 일반화 성능을 보임을 확인했습니다.
  <img width="848" height="188" alt="image" src="https://github.com/user-attachments/assets/4480f890-7ded-4cd4-a192-e3a8e86ff724" />
  
