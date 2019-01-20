#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Dense(units, input_matrix, labels, epochs):
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(110)
    #Initialization in Glorot's Form
    W2 = np.random.randn(input_matrix.shape[1],16)/np.sqrt(input_matrix.shape[1])
    W3 = np.random.randn(16,8)/np.sqrt(16)
    W4 = np.random.randn(8,4)/np.sqrt(8)
    W5 = np.random.randn(4,units)/np.sqrt(4)
    y = labels
    
    avg_loss_list = []
    weights = {}
    epochs_list = [i for i in range(epochs)]
    
    #Parameters for RmsProp
    epsilon = 1e-4
    decay_rate = 0.0
    cache2,cache3,cache4,cache5 = 0,0,0,0
    
    for i in range(1,epochs+1):
        #Forward Pass
        Z1 = input_matrix
        A1 = Z1
        Z2 = np.dot(A1,W2)
        A2 = Z2
        Z3 = np.dot(A2,W3)
        A3 = Z3
        Z4 = np.dot(A3,W4)
        A4 = 1/(1 + np.exp(-Z4))
        Z5 = np.dot(A4,W5)
        A5 = 1/(1 + np.exp(-Z5))
        y_pred = A5
       
        Loss = np.mean((y - y_pred)**2)
        avg_loss_list.append(Loss)
        
        #Backward Pass
        dL_dy_pred = -2*(y - y_pred)
        layer5_delta = np.multiply(dL_dy_pred, A5*(1 - A5))
        layer4_delta = np.multiply(np.dot(layer5_delta, W5.T), A4*(1 - A4))
        layer3_delta = np.multiply(np.dot(layer4_delta, W4.T), 1)
        layer2_delta = np.multiply(np.dot(layer3_delta, W3.T), 1)
        dL_dW5 = np.dot(A4.T, layer5_delta)
        dL_dW4 = np.dot(A3.T, layer4_delta)
        dL_dW3 = np.dot(A2.T, layer3_delta)
        dL_dW2 = np.dot(A1.T, layer2_delta)
        cache5 += (decay_rate*cache5)+(1 - decay_rate)*(dL_dW5)**2
        cache4 += (decay_rate*cache4)+(1 - decay_rate)*(dL_dW4)**2
        cache3 += (decay_rate*cache3)+(1 - decay_rate)*(dL_dW3)**2
        cache2 += (decay_rate*cache2)+(1 - decay_rate)*(dL_dW2)**2
        
        #Upadating Weights using SGD
        learning_rate = 0.01
        W5 -= learning_rate*dL_dW5/(np.sqrt(cache5) + epsilon)
        W4 -= learning_rate*dL_dW4/(np.sqrt(cache4) + epsilon)
        W3 -= learning_rate*dL_dW3/(np.sqrt(cache3) + epsilon)
        W2 -= learning_rate*dL_dW2/(np.sqrt(cache2) + epsilon)
        
        #Loss and Accuracy measure
        if (i%100 == 0):
            count = 0
            for j in range(y.shape[0]):
                if (np.argmax(y_pred, axis = 1)[j] != np.argmax(y, axis = 1)[j]):
                    count += 1
            Accuracy = 1 - (count/y.shape[0])
            print('Epoch:: {}, Loss = {}, Accuracy = {} %'.format(i,Loss,Accuracy*100))
    
    #Storing prediction and Updated final weights
    y_prediction = np.argmax(y_pred, axis = 1)
    weights['W2'], weights['W3'], weights['W4'], weights['W5'] = W2, W3, W4, W5
    plt.plot(epochs_list, avg_loss_list)
    plt.title('Loss vs Epoch')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
    return y_prediction, weights


def evaluate(test_input, test_label, weight_dict):
    import numpy as np
    #Forward Pass
    Z1 = test_input
    A1 = Z1
    Z2 = np.dot(A1,weight_dict['W2'])
    A2 = Z2
    Z3 = np.dot(A2,weight_dict['W3'])
    A3 = Z3
    Z4 = np.dot(A3,weight_dict['W4'])
    A4 = 1/(1 + np.exp(-Z4))
    Z5 = np.dot(A4,weight_dict['W5'])
    A5 = 1/(1 + np.exp(-Z5))
    y_pred = A5
    
    y_prediction = np.argmax(y_pred, axis = 1)
    count = 0
    for j in range(test_label.shape[0]):
        if (np.argmax(y_pred, axis = 1)[j] != np.argmax(test_label, axis = 1)[j]):
            count += 1
    Accuracy = 1 - (count/test_label.shape[0])
    print('Actual Labels = \n'+str(np.argmax(test_label, axis = 1)))
    print('Predicted Labels = \n'+str(y_prediction))
    print('Test Accuracy = {} % '.format(Accuracy*100))
    print('Misclassified label count = '+str(count))

