def test_with_random_noise(test_data, network, num_points):
    network.eval()
    set_seed()
    
    with torch.no_grad():
        num_correct = 0.0
        
        for data, y_actual in test_data:
            if isinstance(network, NonConvModel):
                data = data.view(data.shape[0], -1)
                
                # Choose n random pixel to perturb
                num_examples = data.shape[0]
                pixel_loc = np.random.randint(data.shape[0]-1, size=(num_examples,num_points))
                pixel_val = np.random.rand(num_examples, num_points)

                for j in range(num_examples):
                    for i in range(num_points):
                        p_loc = pixel_loc[j,i]
                        p_val = pixel_val[j,i]

                        data[j,p_loc] = p_val
            else:
                num_examples = data.shape[0]
                num_rows = data.shape[-2]
                num_cols = data.shape[-1]
                
                # Choose n random pixel to perturb
                pixel_row_loc = np.random.randint(num_rows-1, size=(num_examples,num_points))
                pixel_col_loc = np.random.randint(num_cols-1, size=(num_examples,num_points))
                pixel_val = np.random.rand(num_examples, num_points)

                for j in range(num_examples):
                    for i in range(num_points):
                        p_row = pixel_row_loc[j, i]
                        p_col = pixel_col_loc[j, i]
                        p_val = pixel_val[j,i]
                        
                        data[j,:,p_row,p_col] = p_val
                
            # Retrieve log probabilities of class labels
            y_pred = network(data)
            
            # Identify max prediction value
            y_max_pred = y_pred.data.max(1, keepdim=True)[1]
            
            # Count the number correct
            num_correct += y_max_pred.eq(y_actual.data.view_as(y_max_pred)).sum()

        return (float(num_correct) / len(test_data.dataset))
    
def evaluate_random_noise(test_data, net):
    base_acc = test_with_random_noise(test_data, net, 0)
    accuracies = [base_acc]
    num_noisy_points = [0]
    for i in range(1,input_size//2, 10):
        acc = test_with_random_noise(test_data, net, i)
        accuracies.append(acc)
        num_noisy_points.append(i)
        
    fig, ax = plt.subplots()
    ax.plot(num_noisy_points, accuracies) 
    ax.set_xlabel('Number of Noisy Points')
    ax.set_ylabel('Accuracy (%)')
    if isinstance(net, ConvModel):
        ax.set_title('Convolutional Classification')
    else:
        ax.set_title('Non-Convolutional Classification')

    plt.show()
    
evaluate_random_noise(test_loader, nonconv_net)
evaluate_random_noise(test_loader, conv_net)