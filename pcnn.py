import tensorflow
from keras.layers import Conv2D, BatchNormalization, Dropout, ReLU, Dense
from keras import Model, Sequential

class pcnn(Model):
    def __init__(self):
        super(pcnn, self).__init__()
        self.conv1 = Conv2D(filters=2, kernel_size=2, padding='SAME')
        self.bn1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

        self.conv2 = Conv2D(filters=2, kernel_size=2, padding='SAME')
        self.bn2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

        self.conv3 = Conv2D(filters=2, kernel_size=2, padding='SAME')
        self.bn3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

    def convconn(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=False)
        x = ReLU(x)

        x = self.conv2(x)
        x = self.bn2(x, training=False)
        x = ReLU(x)

        x = self.conv3(x)
        x = self.bn3(x, training=False)
        x = ReLU(x)

        return x

def fcconn():
    model = Sequential(
        [
            Dense(32, activation='relu'),
            Dropout(rate=0.02),
            Dense(16, activation='softmax')
        ]
    )
    return model