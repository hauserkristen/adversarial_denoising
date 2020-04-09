import torch
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models import ConvClassificationModel
from data import get_data

def plot_filter_layer(filter_layers):
    # Visualize each filter
    for filter_index, filter_layer in enumerate(filter_layers):
        num_output_channels = filter_layer.shape[0]
        num_input_channels = filter_layer.shape[1]

        # Create figure
        fig = make_subplots(
            rows=num_output_channels, 
            cols=num_input_channels,
            subplot_titles=['Conv Layer: {}<br>I/O Channel: {}/{}'.format(filter_index, i, o)  for o in range(num_output_channels) for i in range(num_input_channels)]
        )

        for o in range(num_output_channels):
            for i in range(num_input_channels):
                # Get filter
                vis_filter = filter_layer[o,i,:,:]

                # Plot
                fig.add_trace(
                    go.Heatmap(
                        z=vis_filter,
                        type='heatmap', 
                        coloraxis='coloraxis',
                        showscale=False
                    ),
                    row=o+1,
                    col=i+1
                )
                axis_num = (o*num_input_channels + i) + 1
                fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num), row=o+1, col=i+1)
                fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=o+1, col=i+1)
        
        fig.update_layout(
            autosize=False,
            height=300*num_output_channels,
            coloraxis={
                'colorscale': 'Gray'
            }
        )
        fig.show()
        input()

def visualize_filters():
    # Create models
    conv_net = ConvClassificationModel()

    # Load model
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Plot convolutional filters
    filters = [
        conv_net.conv1.weight.detach().numpy(),
        conv_net.conv2.weight.detach().numpy()
    ]
    plot_filter_layer(filters)