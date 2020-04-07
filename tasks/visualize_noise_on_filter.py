import torch
import numpy as np
import torch.nn.functional as F

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data


def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)


def get_plots(data, conv_net, display_type, filter_index):
    trans_data = conv_net._transform_input(data)

    layer_1_labels = ['Channel: 1', 'Channel: 2', 'Channel: 3']

    # After first convolution
    after_conv1 = conv_net.conv1(trans_data)
    if display_type == 'Filter' and filter_index == 1:
        return after_conv1.detach().numpy()[0,:,:,:], layer_1_labels

    # After first acivation
    after_act1 = F.relu(F.max_pool2d(after_conv1, 2, 2))
    if display_type == 'Activation' and filter_index == 1:
        return after_act1.detach().numpy()[0,:,:,:], layer_1_labels

    layer_2_labels = ['Channel: 1', 'Channel: 2', 'Channel: 3', 'Channel: 4', 'Channel: 5']

    # After second convolution
    after_conv2 = conv_net.conv2(after_act1)
    if display_type == 'Filter' and filter_index == 2:
        return after_conv2.detach().numpy()[0,:,:,:], layer_2_labels

    # After second acivation
    after_act2 = F.relu(F.max_pool2d(after_conv2, 2, 2))
    return after_act2.detach().numpy()[0,:,:,:], layer_2_labels
    

def visualize_noisy_affects_filter():
    # Hyper parameters
    seed = 2
    data_name = 'MNIST'

    # Download MNIST data set
    set_seed(seed)
    test_set = get_data(data_name, False)
    test_set_n = get_data(data_name, False, 'snp', 0.1)

    # MNIST digit dataset values
    input_size = np.prod(test_set.data.shape[1:])
    output_size = len(test_set.classes)

    # Create models
    conv_net = ConvClassificationModel()

    # Load models
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Create dash app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Label('Test Set Image Index:'),
                dcc.Dropdown(
                    id='data_set-index',
                    options=[{'label': i, 'value': i} for i in range(len(test_set))],
                    value=0
                )
            ],
            style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.Label('Network Layer:'),
                dcc.Dropdown(
                    id='filter-index',
                    options=[{'label': i, 'value': i} for i in  [1,2]],
                    value=1
                ),
            ],
            style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.Label('Network Layer Type:'),
                dcc.RadioItems(
                    id='display-type',
                    options=[{'label': i, 'value': i} for i in ['Filter', 'Activation']],
                    value='Filter',
                    labelStyle={'display': 'inline-block'}
                )
            ],
            style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'})
        ]),
        html.Div([
            html.Div([
                html.Div(id='original-label'),
                html.Div(id='clean-label'),
                html.Div(id='noisy-label'),
                dcc.Graph(id='raw-graph')
            ],
            style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                dcc.Graph(id='filter-graph')
            ],
            style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'})
        ])
    ])

    @app.callback(
    Output('filter-graph', 'figure'),
    [Input('data_set-index', 'value'),
    Input('display-type', 'value'),
    Input('filter-index', 'value')])
    def update_graph(selected_index, display_type, filter_index):
        clean_data, clean_label = test_set[selected_index]
        noisy_data, noisy_label = test_set_n[selected_index]

        clean_data = clean_data.view(1,*clean_data.size()).float()
        noisy_data = noisy_data.view(1,*noisy_data.size()).float()

        clean_plots, labels = get_plots(clean_data, conv_net, display_type, filter_index)
        noisy_plots, _ = get_plots(noisy_data, conv_net, display_type, filter_index)

        num_rows = clean_plots.shape[0]
        fig = make_subplots(
            rows=num_rows, 
            cols=3,
            subplot_titles=['{} on {}'.format(t, l) for l in labels for t in ['Clean Data', 'Noisy Data', 'Difference']])

        for i in range(num_rows):
            clean_plot = np.flipud(clean_plots[i,:,:])
            noisy_plot = np.flipud(noisy_plots[i,:,:])
            diff_plot = clean_plot - noisy_plot

            axis_num = (i*3) + 1

            fig.add_trace(
                go.Heatmap(
                    z=clean_plot,
                    type='heatmap', 
                    coloraxis='coloraxis',
                    showscale=False
                ),
                row=i+1,
                col=1
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num), row=i+1, col=1)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=1)

            fig.add_trace(
                go.Heatmap(
                    z=noisy_plot,
                    type='heatmap', 
                    coloraxis='coloraxis',
                    showscale=False
                ),
                row=i+1,
                col=2
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num+1), row=i+1, col=2)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=2)

            fig.add_trace(
                go.Heatmap(
                    z=diff_plot,
                    type='heatmap', 
                    coloraxis='coloraxis',
                    showscale=False
                ),
                row=i+1,
                col=3
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num+2), row=i+1, col=3)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=3)

        fig.update_layout(
            autosize=False,
            height=300*num_rows,
            coloraxis={
                'colorscale': 'Gray'
            }
        )

        return fig

    @app.callback(
    [Output('raw-graph', 'figure'),
    Output('original-label', 'children'),
    Output('clean-label', 'children'),
    Output('noisy-label', 'children')],
    [Input('data_set-index', 'value')])
    def update_graph(selected_index):
        clean_data, original_label = test_set[selected_index]
        noisy_data, _ = test_set_n[selected_index]

        clean_data_torch = clean_data.view(1, *clean_data.shape)
        noidy_data_torch = noisy_data.view(1, *noisy_data.shape)

        clean_label = conv_net.get_label(clean_data_torch).detach().numpy()[0][0]
        noisy_label = conv_net.get_label(noidy_data_torch).detach().numpy()[0][0]

        clean_data_np = np.flipud(clean_data.detach().numpy()[0,:,:])
        noisy_data_np = np.flipud(noisy_data.detach().numpy()[0,:,:])

        fig = make_subplots(
            rows=1, 
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=['Clean Image', 'Noisy Image'])

        fig.add_trace(
            go.Heatmap(
                z=clean_data_np,
                type='heatmap', 
                coloraxis='coloraxis',
                showscale=False
            ),
            row=1,
            col=1
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y1', row=1, col=1)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=1)

        fig.add_trace(
            go.Heatmap(
                z=noisy_data_np,
                type='heatmap', 
                coloraxis='coloraxis',
                showscale=False
            ),
            row=1,
            col=2
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y2', row=1, col=2)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)

        fig.update_layout(
            autosize=False,
            coloraxis={
                'colorscale': 'Gray',
                'showscale': False
            }
        )

        return fig, 'Original Label: {}'.format(original_label), 'Clean Label: {}'.format(clean_label), 'Noisy Label: {}'.format(noisy_label)

    app.run_server(debug=True)