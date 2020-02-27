from tasks import train_mnist_digit_models, visualize_filters, visualize_noisy_affects

def main():
    # Train the initial models
    #train_mnist_digit_models()

    # Visualize the learned filters
    visualize_filters()

    # Visualize noisy image effects on filters
    visualize_noisy_affects()

if __name__ == '__main__':
    main()