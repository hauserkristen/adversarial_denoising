import torch
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data
from .common import set_seed, get_plots, calculate_similarity

def search_for_adv_filter_match():
    # Parameters
    seed = 2
    data_name = 'MNIST'
    adv_image_index = 20
    
    # Create models
    set_seed(seed)
    conv_net = ConvClassificationModel()

    # Load models
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Read noisy image file
    filename = 'data\\{}\\adv_data_np\\{}_{}_clean.npy'.format(data_name, 'FGSM', adv_image_index)
    clean_data = np.flipud(np.load(filename)).copy()

    # Get second feature representation for index
    clean_data_torch = torch.from_numpy(clean_data)
    clean_data_torch = clean_data_torch.view(1, 1, *clean_data_torch.shape).float()
    clean_label = conv_net.get_label(clean_data_torch).detach().numpy()[0][0]
    clean_second_feature_set = get_plots(clean_data_torch, conv_net)[(2, 'Filter')][0]

    # Read noisy image file
    filename = 'data\\{}\\adv_data_np\\{}_{}_adv.npy'.format(data_name, 'FGSM', adv_image_index)
    noisy_data = np.flipud(np.load(filename)).copy()
   
    # Get second feature representation for index
    noisy_data_torch = torch.from_numpy(noisy_data)
    noisy_data_torch = noisy_data_torch.view(1, 1, *noisy_data_torch.shape).float()
    noisy_label = conv_net.get_label(noisy_data_torch).detach().numpy()[0][0]
    noisy_second_feature_set = get_plots(noisy_data_torch, conv_net)[(2, 'Filter')][0]

    # Get training data set
    train_set = get_data(data_name, True)
    
    # Search for similar feature set
    sim_feature_clean = ()
    sim_feature_noisy = ()
    max_sim_clean = -float('inf')
    max_sim_noisy= -float('inf')
    for i in range(len(train_set)):
        train_data, train_label = train_set[i]

        if train_label == noisy_label or train_label == clean_label:
            # Compare second feature set
            train_data_torch = train_data.view(1, *train_data.shape).float()
            train_second_feature_set = get_plots(train_data_torch, conv_net)[(2, 'Filter')][0]

            simularity = 0.0
            num_features = train_second_feature_set.shape[0]

            if train_label == noisy_label:
                for j in range(num_features):
                    # Calculate similarity
                    simularity += calculate_similarity(train_second_feature_set[j,:,:], noisy_second_feature_set[j,:,:])
                simularity /= float(num_features)

                if simularity > max_sim_noisy:
                    sim_feature_noisy = (simularity, i, train_data, train_second_feature_set)
                    max_sim_noisy = simularity
            elif train_label == clean_label:
                for j in range(num_features):
                    # Calculate similarity
                    simularity += calculate_similarity(train_second_feature_set[j,:,:], clean_second_feature_set[j,:,:])
                simularity /= float(num_features)

                if simularity > max_sim_clean:
                    sim_feature_clean = (simularity, i, train_data, train_second_feature_set)
                    max_sim_clean = simularity


    filter_fig = make_subplots(
        rows=num_features, 
        cols=4,
        subplot_titles=['Similar Clean<br>Class Training<br>Image', 'Clean Image', 'Adversarial Image', 'Similar Adversarial<br>Class Training<br>Image'] + ['']*((num_features-1)*4))

    for i in range(num_features):
        noisy_plot = np.flipud(noisy_second_feature_set[i,:,:])
        clean_plot = np.flipud(clean_second_feature_set[i,:,:])
        sim_clean_plot = np.flipud(sim_feature_clean[3][i,:,:])
        sim_noisy_plot = np.flipud(sim_feature_noisy[3][i,:,:])

        axis_num = (i*4) + 1

        filter_fig.add_trace(
            go.Heatmap(
                z=sim_clean_plot,
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

        filter_fig.add_trace(
            go.Heatmap(
                z=noisy_plot,
                type='heatmap', 
                coloraxis='coloraxis',
                showscale=False
            ),
            row=i+1,
            col=3
        )
        filter_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num+2), row=i+1, col=3)
        filter_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=3)

        filter_fig.add_trace(
            go.Heatmap(
                z=sim_noisy_plot,
                type='heatmap', 
                coloraxis='coloraxis',
                showscale=False
            ),
            row=i+1,
            col=4
        )
        filter_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y{}'.format(axis_num+3), row=i+1, col=4)
        filter_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i+1, col=4)

    filter_fig.update_layout(
        autosize=False,
        height=300*num_features,
        coloraxis={
            'colorscale': 'Gray'
        }
    )

    filter_fig.show()

    raw_fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=['Clean Image', 'Adversarial Image', 'Similar Clean<br>Class Training<br>Image', 'Similar Adversarial<br>Class Training<br>Image'])

    raw_fig.add_trace(
        go.Heatmap(
            z=np.flipud(clean_data),
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
            z=np.flipud(noisy_data),
            type='heatmap', 
            coloraxis='coloraxis',
            showscale=False
        ),
        row=1,
        col=2
    )
    raw_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y2', row=1, col=2)
    raw_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)

    raw_fig.add_trace(
        go.Heatmap(
            z=np.flipud(sim_feature_clean[2].detach().numpy()[0,:,:]),
            type='heatmap', 
            coloraxis='coloraxis',
            showscale=False
        ),
        row=2,
        col=1
    )
    raw_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y3', row=2, col=1)
    raw_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=2, col=1)

    raw_fig.add_trace(
        go.Heatmap(
            z=np.flipud(sim_feature_noisy[2].detach().numpy()[0,:,:]),
            type='heatmap', 
            coloraxis='coloraxis',
            showscale=False
        ),
        row=2,
        col=2
    )
    raw_fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, scaleanchor='y4', row=2, col=2)
    raw_fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=2, col=2)

    raw_fig.update_layout(
        autosize=False,
        coloraxis={
            'colorscale': 'Gray',
            'showscale': False
        }
    )

    raw_fig.show()
    print('Done')