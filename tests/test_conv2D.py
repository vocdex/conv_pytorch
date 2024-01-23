import pytest
import torch
from src import CustomConv2D


@pytest.fixture
def conv_params():
    """
    Fixture for creating predefined set of parameters for a convolutional layer.

    Returns:
    A dictionary of parameters for a convolutional layer.
    """
    return {
        "batch_size": 8,
        "in_channels": 3,
        "height": 16,
        "width": 16,
        "out_channels": 64,
        "kernel_size": (3, 3),
        "padding": (0, 0),
    }

class TestCustomConv2D:
    def test_forward(self, conv_params):
        """
        Test the forward pass of the convolutional layer.

        Arguments:
        conv_params: Dictionary of parameters for a convolutional layer.
        """

        conv = CustomConv2D(
            in_channels=conv_params["in_channels"],
            out_channels=conv_params["out_channels"],
            kernel_size=conv_params["kernel_size"],
            padding=conv_params["padding"],
        )
        input = torch.randn(
            conv_params["batch_size"],
            conv_params["in_channels"],
            conv_params["height"],
            conv_params["width"],
        )
        print(input.shape)
        output = conv(input)
        assert output.shape == (
            conv_params["batch_size"],
            conv_params["out_channels"],
            conv.get_output_size(conv_params["height"]),
            conv.get_output_size(conv_params["width"]),
        )
    