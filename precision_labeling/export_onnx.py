import lightning as L
import torch
from torch import nn


class SimpleCNN(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = False
        self.save_hyperparameters()

        # very big
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # medium
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # big
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # small
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, 2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        input_img = input_img / 2500.0  # preprocess inside forward for easy onnx export
        x = self.layers(input_img)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


model_path = "mlruns/898661618971256471/588188e26e1942689f2dc91587da30cc/checkpoints/epoch=675-step=25688.ckpt"
model = SimpleCNN.load_from_checkpoint(model_path).to("cpu")
model.eval()
example_inputs = torch.randn([1, 1, 224, 224]).to("cpu")
program = torch.onnx.export(
    model,
    example_inputs,
    dynamo=True,
    input_names=["input"],
    output_names=["output"],
)
program.save("medium.onnx")
