import os
import torch
import numpy as np
from bisect import bisect
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as plt_offline

from .common import load_model_and_data, format_torch
from .DAG import DAG

NOISE_CONFIGS = [
    ('gaussian', 50, 'Gaussian<br>&#956;=0, &#963;=50'),
    ('gaussian', 100, 'Gaussian<br>&#956;=0, &#963;=100'),
    ('poisson', 100, 'Poisson<br>&#956;=100, dispersion=75'),
    ('poisson', 200, 'Poisson<br>&#956;=200, dispersion=75'),
    ('poisson', 500, 'Poisson<br>&#956;=500, dispersion=75'),
    ('impulse', 0.2, 'Impulse<br>p=0.2'),
    ('impulse', 0.5, 'Impulse<br>p=0.5')
]

def display_noise_histograms(image_index, results):
    titles = [r[0] for r in results]
    titles.insert(6, '')
    titles.append('')

    fig = make_subplots(
        rows=3, 
        cols=3,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=titles
    )

    for i, r in enumerate(results):
        noise_title, orig_np, noisy_np, denoised_np, adv_image_np, adv_denoised_np = r

        # Calculate difference
        noise_np = noisy_np - orig_np
        adv_noise_np = adv_image_np - orig_np


        row = (i // 3) + 1
        col = (i % 3) + 1

        if row == 1 and col == 1:
            show_legend = True
        elif row == 3 and col == 1:
            col = 2
            show_legend = False
        else:
            show_legend = False

        # Create histogram of noise distribution
        noise_vals = []
        adversarial_vals = []
        for j in range(noise_np.shape[0]):
            for k in range(noise_np.shape[1]):
                for l in range(noise_np.shape[2]):
                    noise_vals.append(noise_np[j,k,l])
                    adversarial_vals.append(adv_noise_np[j,k,l])

        # Add traces
        fig.add_trace(
            go.Histogram(
                x=noise_vals,
                nbinsx=25,
                name='Noise',
                marker_color='#1f77b4',
                showlegend=show_legend
            ),
            row=row,
            col=col
        )
        fig.add_trace(
            go.Histogram(
                x=adversarial_vals,
                nbinsx=25,
                name='Adversarial',
                marker_color='#ff7f0e',
                showlegend=show_legend
            ),
            row=row,
            col=col
        )

    # Format figure
    fig.update_layout(
        bargap=0.15,
        legend={
            'x': 0.15, 
            'y': 0.15
        }
    )

    # Create save directory name
    full_directory = 'images//final_adv_examples'

    # Create directory if required
    if not os.path.exists(full_directory):
        os.mkdir(full_directory)

    # Save figure
    filename = '{}//histogram_{}.html'.format(full_directory, image_index)
    plt_offline.plot(fig, filename=filename, auto_open=False)

def display_images(image_index, results):
    num_results = len(results)
    titles = ['Noisy Image', 'Denoised Image', 'Adversarial<br>Noisy Image', 'Denoised<br>Adversarial Image'] + ['']*((num_results-1)*4)

    fig = make_subplots(
        rows=num_results, 
        cols=4,
        horizontal_spacing=0.005,
        vertical_spacing=0.005,
        subplot_titles=titles
    )

    annotation_interval_y = [0.95,   0.8,   0.65,   0.5,  0.35,   0.2,  0.05]
    annotation_interval_x = [-0.065, -0.065, -0.11, -0.11, -0.11, -0.065, -0.065]
    for i, r in enumerate(results):
        noise_title, orig_np, noisy_np, denoised_np, adv_image_np, adv_denoised_np = r

        row = i + 1

        plots = [
            noisy_np, denoised_np,
            adv_image_np, adv_denoised_np
        ]

        for col, p in enumerate(plots):
            fig.add_trace(
                go.Image(
                    z=p,
                    colormodel='rgb',
                    zmax=[1,1,1,1],
                    zmin=[0,0,0,0]
                ),
                row=row,
                col=col+1
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col+1)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col+1)

        fig.add_annotation(
            x=annotation_interval_x[i],
            y=annotation_interval_y[i],
            text=noise_title,
            textangle=0
        )

    fig.update_annotations(xref='paper', yref='paper', showarrow=False)

    ## Create save directory name
    full_directory = 'images//final_adv_examples'

    # Create directory if required
    if not os.path.exists(full_directory):
        os.mkdir(full_directory)

    # Save figure
    filename = '{}//images_{}.html'.format(full_directory, image_index)
    plt_offline.plot(fig, filename=filename, auto_open=False)

def visualize_final_examples():
    # Set choosen index
    num_samples = 25
    selected_index = 1099

    # Get images for every noise config
    results = []
    for noise_type, noise_param, noise_title in NOISE_CONFIGS:
        # Load data
        net, test_set_original, test_set_noisy = load_model_and_data(noise_type, noise_param)

        # Randomly choose indices
        num_examples = len(test_set_noisy)
        save_indices = np.random.randint(num_examples, size=num_samples)

        # Test
        for i in save_indices:
            orig_data, orig_label = test_set_original[i]
            noisy_data, _ = test_set_noisy[i]

            # Proper format
            orig_data = orig_data.unsqueeze(0).float()
            noisy_data = noisy_data.unsqueeze(0).float()

            # Denoise
            denoised_result = net(noisy_data)
            denoised_result = format_torch(denoised_result)

            # Call attack
            adversarial_noise = DAG(net, orig_data, noisy_data)
            adversarial_data = orig_data + adversarial_noise
            adversarial_data = format_torch(adversarial_data)

            # Denoise
            adv_denoised_result = net(adversarial_data)
            adv_denoised_result = format_torch(adv_denoised_result)

            if i == selected_index:
                # Post process
                orig_np = orig_data.detach().numpy().squeeze(0)
                noisy_np = noisy_data.detach().numpy().squeeze(0)
                denoised_np = denoised_result.detach().numpy().squeeze(0)
                adv_image_np = adversarial_data.detach().numpy().squeeze(0)
                adv_denoised_np = adv_denoised_result.detach().numpy().squeeze(0)

                # Flip axes back
                orig_np = np.moveaxis(orig_np, 0, -1)
                noisy_np = np.moveaxis(noisy_np, 0, -1)
                denoised_np = np.moveaxis(denoised_np, 0, -1)
                adv_image_np = np.moveaxis(adv_image_np, 0, -1)
                adv_denoised_np = np.moveaxis(adv_denoised_np, 0, -1)

                # Display and save image
                results.append(
                    (noise_title, orig_np, noisy_np, denoised_np, adv_image_np, adv_denoised_np)
                )
                break

    display_images(selected_index, results)

    display_noise_histograms(selected_index, results)