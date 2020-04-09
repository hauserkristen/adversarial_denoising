import torch
import numpy as np
from skimage.metrics import structural_similarity

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data
from .common import set_seed, get_plots

def calculate_similarity(clean_image, noisy_image):
    #Structural similarity (SSIM) index
    score = structural_similarity(noisy_image, clean_image, data_range=(noisy_image.max() - noisy_image.min()))
    return score

def search_for_filter_match():
    # Parameters
    seed = 2
    data_name = 'MNIST'
    show_all = False
    noise_type = 'snp'
    noisy_image_index = 175

    # Download MNIST data set
    set_seed(seed)
    train_set = get_data(data_name, True)
    test_set_n = get_data(data_name, False, noise_type, 0.1)

    # Create models
    conv_net = ConvClassificationModel()

    # Load models
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Get second feature representation for index
    noisy_data, original_label = test_set_n[noisy_image_index]
    noisy_data_torch = noisy_data.view(1, *noisy_data.shape).float()
    noisy_label = conv_net.get_label(noisy_data_torch).detach().numpy()[0][0]
    noisy_second_feature_set = get_plots(noisy_data_torch, conv_net)[(2, 'Filter')][0]

    # Search for similar clean feature set
    similar_feature_sets = []
    max_sim = -float('inf')
    for i in range(len(train_set)):
        clean_data, clean_label = train_set[i]

        if clean_label == noisy_label:
            # Compare second feature set
            clean_data_torch = clean_data.view(1, *clean_data.shape).float()
            clean_second_feature_set = get_plots(clean_data_torch, conv_net)[(2, 'Filter')][0]

            simularity = 0.0
            num_features = clean_second_feature_set.shape[0]
            for j in range(num_features):
                # Calculate similarity
                simularity += calculate_similarity(clean_second_feature_set[j,:,:], noisy_second_feature_set[j,:,:])
            simularity /= float(num_features)

            if simularity > 0.7:
                data = (simularity, i, clean_data, clean_second_feature_set)
                similar_feature_sets.append(data)
            if simularity > max_sim:
                max_sim = simularity

    print('Max Sim Score: {}'.format(max_sim))

    # Create dash app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        html.Div([
            html.Label('Test Set Image Index:'),
            dcc.Dropdown(
                id='data_set-index',
                options=[{'label': i, 'value': i} for i in range(len(similar_feature_sets))],
                value=0
            )
        ]),
        html.Div([
            html.Div([
                html.Div(id='sim-score'),
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
    [Output('filter-graph', 'figure'),
    Output('raw-graph', 'figure'),
    Output('sim-score', 'children')],
    [Input('data_set-index', 'value')])
    def update_graph(selected_index):
        sim_score, data_set_index, clean_data, clean_second_feature_set = similar_feature_sets[selected_index]

        filter_fig = make_subplots(
            rows=num_features, 
            cols=2,
            subplot_titles=['{} on {}'.format(t, l) for l in ['Channel: 1', 'Channel: 2', 'Channel: 3', 'Channel: 4', 'Channel: 5'] for t in ['Noisy Image', 'Similar Clean Training Image']])

        for i in range(num_features):
            noisy_plot = np.flipud(noisy_second_feature_set[i,:,:])
            clean_plot = np.flipud(clean_second_feature_set[i,:,:])

            axis_num = (i*2) + 1

            filter_fig.add_trace(
                go.Heatmap(
                    z=noisy_plot,
                    type='heatmap', 
                    coloraxis='coloraxis',
                    showscale=False
                ),
                row=i+1,
                col=1
            )
            filter_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num), row=i+1, col=1)
            filter_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=1)

            filter_fig.add_trace(
                go.Heatmap(
                    z=clean_plot,
                    type='heatmap', 
                    coloraxis='coloraxis',
                    showscale=False
                ),
                row=i+1,
                col=2
            )
            filter_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num+1), row=i+1, col=2)
            filter_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=2)

        filter_fig.update_layout(
            autosize=False,
            height=300*num_features,
            coloraxis={
                'colorscale': 'Gray'
            }
        )

        raw_fig = make_subplots(
            rows=1, 
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=['Noisy Image', 'Similar Clean Training Image'])

        raw_fig.add_trace(
            go.Heatmap(
                z=np.flipud(noisy_data.detach().numpy()[0,:,:]),
                type='heatmap', 
                coloraxis='coloraxis',
                showscale=False
            ),
            row=1,
            col=1
        )
        raw_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y1', row=1, col=1)
        raw_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=1)

        raw_fig.add_trace(
            go.Heatmap(
                z=np.flipud(clean_data.detach().numpy()[0,:,:]),
                type='heatmap', 
                coloraxis='coloraxis',
                showscale=False
            ),
            row=1,
            col=2
        )
        raw_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y2', row=1, col=2)
        raw_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)

        raw_fig.update_layout(
            autosize=False,
            coloraxis={
                'colorscale': 'Gray',
                'showscale': False
            }
            )

        return filter_fig, raw_fig, 'Similarity Score: {}'.format(sim_score)

    app.run_server(debug=True)