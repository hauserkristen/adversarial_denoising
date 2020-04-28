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
from tasks import set_seed, get_plots

def visualize_noisy_affects_filter():
    # Parameters
    seed = 2
    data_name = 'MNIST'
    show_all = False
    noise_type = 'gaussian_gray'

    # Save indices
    if noise_type == 'snp':
        save_indices = [118, 175]
    else:
        save_indices = [333, 1444]

    # Download MNIST data set
    set_seed(seed)
    test_set_n = get_data(data_name, False, noise_type=noise_type, percent_noise=0.1)
    test_set = get_data(data_name, False)
    
    # Create models
    conv_net = ConvClassificationModel()

    # Load models
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Pre-compute values
    raw_results = {}
    filter_results = {}
    for i in range(len(test_set)):
        clean_data, original_label = test_set[i]
        clean_data_torch = clean_data.view(1, *clean_data.shape).float()
        clean_label = conv_net.get_label(clean_data_torch).detach().numpy()[0][0]
        clean_p_dist = np.round(np.exp(conv_net.classify(clean_data_torch).detach().numpy()[0]), decimals=2)
        clean_data_np = np.flipud(clean_data.detach().numpy()[0,:,:])

        noisy_data, _ = test_set_n[i]
        noisy_data_torch = noisy_data.view(1, *noisy_data.shape).float()
        noisy_label = conv_net.get_label(noisy_data_torch).detach().numpy()[0][0]
        noisy_p_dist = np.round(np.exp(conv_net.classify(noisy_data_torch).detach().numpy()[0]), decimals=2)
        noisy_data_np = np.flipud(noisy_data.detach().numpy()[0,:,:])

        label_mismatch = original_label == clean_label and original_label != noisy_label

        if show_all or label_mismatch:
            raw_results[i] = (clean_data_np, noisy_data_np, original_label, clean_label, noisy_label, clean_p_dist, noisy_p_dist)
            filter_results[i] = (get_plots(clean_data_torch, conv_net), get_plots(noisy_data_torch, conv_net))

            if i in save_indices:
                # Dump output
                filename = 'data\\{}\\noisy_data_np\\{}_{}'.format(data_name, noise_type, i)
                np.save(filename, noisy_data_np)

    # Create dash app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Label('Test Set Image Index:'),
                dcc.Dropdown(
                    id='data_set-index',
                    options=[{'label': i, 'value': i} for i in raw_results.keys()],
                    value=list(raw_results.keys())[0]
                )
            ],
            style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.Label('Network Layer:'),
                dcc.Dropdown(
                    id='filter-index',
                    options=[{'label': i, 'value': i} for i in  [1,2]],
                    value=2
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
        clean_plot_set, noisy_plot_set = filter_results[selected_index]

        clean_plots, labels = clean_plot_set[(filter_index, display_type)]
        noisy_plots, _ = noisy_plot_set[(filter_index, display_type)]

        num_rows = clean_plots.shape[0]
        fig = make_subplots(
            rows=num_rows, 
            cols=3,
            subplot_titles=['{} on {}'.format(t, l) for l in labels for t in ['Clean Image', 'Noisy Image', 'Difference']])

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
        clean_data_np, noisy_data_np, original_label, clean_label, noisy_label, clean_p_dist, noisy_p_dist = raw_results[selected_index]

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

        clean_desc = 'Clean Label: {}, Clean Label Probabilities: {}'.format(clean_label, clean_p_dist)
        noisy_desc = 'Noisy Label: {}, Noisy Label Probabilities: {}'.format(noisy_label, noisy_p_dist)

        return fig, 'Original Label: {}'.format(original_label), clean_desc, noisy_desc

    app.run_server(port=8051, debug=True)