import torch


class VGGNet16(torch.nn.Module):
    """
    The VGGNet-16 module.
    """

    def __init__(self, num_classes=1000):

        # Mandatory call to super class module.
        super(VGGNet16, self).__init__()

        b1 = torch.nn.Sequential(
            # Layer 1 - Convolution Layer - Nx3x244x244 -> Nx64x244x244
            torch.nn.Conv2d(in_channels=3, out_channels=64,
                            kernel_size=3),
            torch.nn.ReLU(inplace=True),

            # Layer 2 - Convolution Layer - Nx64x244x244 -> Nx64x244x244
            torch.nn.Conv2d(in_channels=3, out_channels=64,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 3 - Convolution Layer - Nx64x244x244 -> Nx64x112x112
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        b2 = torch.nn.Sequential(

            # Layer 4 - Convolution Layer - Nx64x112x112 -> Nx128x112x112
            torch.nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 5 - Convolution Layer - Nx128x112x112 -> Nx128x112x112
            torch.nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 6 - Convolution Layer - Nx128x112x112 -> Nx128x56x56
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        b3 = torch.nn.Sequential(

            # Layer 7 - Convolution Layer - Nx128x56x56 -> Nx256x56x56
            torch.nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 8 - Convolution Layer - Nx256x56x56 -> Nx256x56x56
            torch.nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 9 - Convolution Layer - Nx256x56x56 -> Nx256x56x56
            torch.nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 11 - Convolution Layer - Nx256x56x56 -> Nx128x28x28
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        b4 = torch.nn.Sequential(

            # Layer 12 - Convolution Layer - Nx128x28x28 -> Nx512x28x28
            torch.nn.Conv2d(in_channels=256, out_channels=512,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 13 - Convolution Layer - Nx512x28x28 -> Nx512x28x28
            torch.nn.Conv2d(in_channels=512, out_channels=512,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 14 - Convolution Layer - Nx512x28x28 -> Nx512x28x28
            torch.nn.Conv2d(in_channels=512, out_channels=512,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 16 - Convolution Layer - Nx512x28x28 -> Nx512x14x14
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        b5 = torch.nn.Sequential(

            # Layer 17 - Convolution Layer - Nx512x14x14 -> Nx512x14x14
            torch.nn.Conv2d(in_channels=512, out_channels=512,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 18 - Convolution Layer - Nx512x14x14 -> Nx512x14x14
            torch.nn.Conv2d(in_channels=512, out_channels=512,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 19 - Convolution Layer - Nx512x14x14 -> Nx512x14x14
            torch.nn.Conv2d(in_channels=512, out_channels=512,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            # Layer 21 - Convolution Layer - Nx512x14x14 -> Nx512x7x7
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        b6 = torch.nn.Sequential(

            # Layer 22 - Fully Connected Layer - Nx1x25088-> Nx1x4096
            torch.nn.Linear(in_features=512*7*7, out_features=4096),
            torch.nn.ReLU(inplace=True),

            # Layer 23 - Fully Connected Layer - Nx1x4096 -> Nx1x4096
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(inplace=True),

            # Layer 24 - Fully Connected Layer - Nx1x4096 -> Nx1xC
            torch.nn.Linear(in_features=4096, out_features=num_classes),
            torch.nn.Softmax(),
        )

        # Defining the feature extraction layers.
        self.feature_extractor = torch.nn.Sequential(b1, b2, b3, b4, b5)

        # Defining the classification layers.
        self.classifier = b6

    def forward(self, x):

        # Forward pass through the feature extractor - Nx3x224x224 -> Nx256x6x6
        x = self.feature_extractor(x)

        # Flattening the feature map - Nx256x6x6 -> Nx1x9216
        x = torch.flatten(x, 1)

        # Forward pass through the classifier - Nx1x9216 -> Nx1xnum_classes
        return self.classifier(x)
