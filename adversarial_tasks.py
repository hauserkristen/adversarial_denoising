from tasks import *

def main():
    # Train the initial models
    #train_mnist_digit_models()

    # Visualize the learned filters
    #visualize_filters()

    # Examine affect of noise on accuracy
    #visualize_noisy_affects_accuracy()

    # Affect of adversarial attacks
    #visualize_adv_affects_accuracy()

    # Visualize noisy image effects on filters
    visualize_noisy_affects_filter()

    # Search for structural similar features
    #search_for_filter_match()

    
if __name__ == '__main__':
    main()