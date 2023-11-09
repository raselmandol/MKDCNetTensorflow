import tensorflow as tf

def conv3x3(filters, stride=1):
    return tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)

def conv1x1(filters, stride=1):
    return tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False)

class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(filters, stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = conv3x3(filters)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, filters, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(filters)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(filters, stride)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = conv1x1(filters * self.expansion)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)

        return out

class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def _make_layer(self, block, filters, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != filters * block.expansion:
            downsample = tf.keras.Sequential([
                conv1x1(filters * block.expansion, stride),
                tf.keras.layers.BatchNormalization()
            ])

        layers = []
        layers.append(block(filters, stride, downsample))
        self.inplanes = filters * block.expansion
        for _ in range(1, blocks):
            layers.append(block(filters))

        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)

        return x

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
