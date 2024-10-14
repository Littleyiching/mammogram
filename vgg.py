from torch import nn
def get_vgg_layers(config, batch_norm):

    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)
class VGG_simple(nn.Module):
    def __init__(self, features, output_dim, feature_size=1024):
        super().__init__()

        self.features = features

        #self.avgpool = nn.AdaptiveAvgPool2d(7)
        #self.avgpool2 = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 16),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(16, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = self.avgpool2(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x

class VGG(nn.Module):
    def __init__(self, features, output_dim, feature_size=1024):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)
        #self.avgpool2 = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, output_dim),
        )
        #self.classifier = nn.Sequential(
        #    nn.Linear(512 * 1 * 1, feature_size),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(0.5),
        #    nn.Linear(feature_size, output_dim),
        #)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #x = self.avgpool2(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x
