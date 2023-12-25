import tensorflow as tf

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, out_c, kernel_size=3, padding='same', dilation=1, use_bias=False, act=True):
        super(Conv2D, self).__init__()
        self.act = act
        self.conv = tf.keras.layers.Conv2D(
            filters=out_c, kernel_size=kernel_size,
            padding=padding, dilation_rate=dilation, use_bias=use_bias
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super(ResidualBlock, self).__init__()
        self.network = tf.keras.Sequential([
            Conv2D(out_c),
            Conv2D(out_c, kernel_size=1, padding='valid', act=False)
        ])
        self.shortcut = Conv2D(out_c, kernel_size=1, padding='valid', act=False)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x + s)
        return x

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.fc1 = tf.keras.layers.Conv2D(in_planes // 16, kernel_size=1, use_bias=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Conv2D(in_planes, kernel_size=1, use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, padding='valid', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=3)
        x = self.conv1(x)
        return self.sigmoid(x)

#encoder B
class Encoder(tf.keras.layers.Layer):
    def __init__(self, ch):
        super(Encoder, self).__init__()
        # Define the layers corresponding to the ResNet backbone
        # You can replace these with your custom ResNet implementation if needed
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_shape=(256, 256, 3)
        )
        self.c1 = Conv2D(ch)
        self.c2 = Conv2D(ch)
        self.c3 = Conv2D(ch)
        self.c4 = Conv2D(ch)

    def call(self, x):
        x0 = x
        x1 = self.backbone.layers[2](x0)
        x2 = self.backbone.layers[4](x1)
        x3 = self.backbone.layers[5](x2)
        x4 = self.backbone.layers[6](x3)
        c1 = self.c1(x1)
        c2 = self.c2(x2)
        c3 = self.c3(x3)
        c4 = self.c4(x4)
        return c1, c2, c3, c4

class MultiKernelDilatedConv(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super(MultiKernelDilatedConv, self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.c1 = Conv2D(out_c, kernel_size=1, act=True)
        self.c2 = Conv2D(out_c, kernel_size=3, act=True)
        self.c3 = Conv2D(out_c, kernel_size=7, act=True)
        self.c4 = Conv2D(out_c, kernel_size=11, act=True)
        self.s1 = Conv2D(out_c, kernel_size=1, act=False)
        self.d1 = Conv2D(out_c, kernel_size=3, dilation=1, act=True)
        self.d2 = Conv2D(out_c, kernel_size=3, dilation=3, act=True)
        self.d3 = Conv2D(out_c, kernel_size=3, dilation=7, act=True)
        self.d4 = Conv2D(out_c, kernel_size=3, dilation=11, act=True)
        self.s2 = Conv2D(out_c, kernel_size=1, act=False)
        self.s3 = Conv2D(out_c, kernel_size=1, act=False)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def call(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        x = self.s1(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        x = self.s2(x)
        s = self.c3(x0)
        x = self.relu(x + s)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class MultiScaleFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super(MultiScaleFeatureFusion, self).__init__()
        self.up2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.up4 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.c1 = Conv2D(out_c)
        self.c2 = Conv2D(out_c)
        self.c3 = Conv2D(out_c)
        self.c4 = Conv2D(out_c)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def call(self, f1, f2, f3):
        x1 = self.up4(f1)
        x2 = self.up2(f2)
        x1 = self.c1(x1)
        x1 = tf.concat([x1, f3], axis=3)
        x2 = self.c2(x2)
        x2 = tf.concat([x2, x1], axis=3)
        x2 = self.up2(x2)
        x2 = self.c4(x2)
        x2 = x2 * self.ca(x2)
        x2 = x2 * self.sa(x2)
        return x2

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super(DecoderBlock, self).__init__()
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.r1 = ResidualBlock(out_c)
        self.r2 = ResidualBlock(out_c)

    def call(self, x, s):
        x = self.up(x)
        x = tf.concat([x, s], axis=3)
        x = self.r1(x)
        x = self.r2(x)
        return x

class DeepSegNet(tf.keras.Model):
    def __init__(self):
        super(DeepSegNet, self).__init__()

        # Rest of your code...
        self.encoder = Encoder(96)
        self.c1 = MultiKernelDilatedConv(96)
        self.c2 = MultiKernelDilatedConv(96)
        self.c3 = MultiKernelDilatedConv(96)
        self.c4 = MultiKernelDilatedConv(96)
        self.d1 = DecoderBlock(96)
        self.d2 = DecoderBlock(96)
        self.d3 = DecoderBlock(96)
        self.msf = MultiScaleFeatureFusion(96)
        self.y = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')

    def call(self, image):
        s0 = image
        s1, s2, s3, s4 = self.encoder(image)
        x1 = self.c1(s1)
        x2 = self.c2(s2)
        x3 = self.c3(s3)
        x4 = self.c4(s4)
        d1 = self.d1(x4, x3)
        d2 = self.d2(d1, x2)
        d3 = self.d3(d2, x1)
        x = self.msf(d1, d2, d3)
        y = self.y(x)
        return y
