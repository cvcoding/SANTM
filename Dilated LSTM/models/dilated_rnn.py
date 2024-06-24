import tensorflow as tf
import math


class DilatedRNN:

    type_to_func_dict = {
        "VanillaRNN": tf.contrib.rnn.BasicRNNCell,
        "LSTM": tf.contrib.rnn.BasicLSTMCell,
        "GRU": tf.contrib.rnn.GRUCell
    }
    available_cell_types = type_to_func_dict.keys()

    def __init__(self, typeof_cell, num_hidden_units, list_of_dilations, dropout = None):

        if typeof_cell not in self.available_cell_types:
            raise ValueError("Valid cell type is 'VanillaRNN' or 'LSTM' or 'GRU'")
        
        self.cell_type = typeof_cell
        self.hidden_unit_size = num_hidden_units

        cell_func = self.type_to_func_dict[typeof_cell]
        self.cell_list = [cell_func(num_hidden_units) for _ in range(len(list_of_dilations))]

        self.list_of_dilations = list_of_dilations

        if dropout is not None:
            self.cell_list[0] = tf.contrib.rnn.DropoutWrapper(cell_list[0], dropout, 1, 1)
            self.dropout = dropout

    
    
    def drnn(self, x_data):
        
        input_data = x_data.copy()
        timesteps = len(input_data)

        
        layer = 1
        
        for dilation, cell_layer in zip(self.list_of_dilations, self.cell_list):
            
            # Define the number of timestamps

            input_data = tf.convert_to_tensor(input_data)
        
            scope = "Layer_%d" %layer

            # Input has shape (T, batch_size, input_size)
            # For dilation d we want to transorm it to shape (T/d, batch_size*d, input_size)

            # Pad the sequence with 0s in order to make T divisible by d
            
            # We want dilation to divide exactly the timestamps
            if (timesteps % dilation == 0):
                # Reduce the sequence length by `dilation` times
                reduced_timesteps = timesteps // dilation
            else:
                reduced_timesteps = math.ceil(timesteps/dilation)
                n_timesteps_to_add = (reduced_timesteps * dilation) - timesteps
                zero_padding = tf.zeros_like(input_data[0])
                zero_padding = tf.tile(tf.expand_dims(zero_padding, axis=0), tf.constant([n_timesteps_to_add, 1, 1]))
                input_data = tf.concat([input_data, zero_padding], axis=0)
                
            input_data = tf.split(input_data, dilation)
            input_data = tf.concat(input_data, axis=1)

            input_data = tf.unstack(input_data)
            reduced_input_to_layer = input_data
            
            reduced_output_from_layer, _ = tf.contrib.rnn.static_rnn(cell_layer, reduced_input_to_layer, dtype=tf.float32, scope = scope)
            
            splitted_tensors = [tf.split(tensor, dilation) for tensor in reduced_output_from_layer]
            output_from_layer = [item for sublist in splitted_tensors for item in sublist]

            input_data = output_from_layer[:timesteps]
            layer += 1
        
        return input_data
    

    def classification(self, input_data, class_num, experiment):
                    
        # Change Tensor's shape from (batch_size, T, input_size) to 
        # list of Tensors with shape (batch_size, input_size)
        # and length of T
        rnn_data = tf.unstack(input_data, axis=1)
        
        outputs = self.drnn(rnn_data)

        if experiment == "cifar":
            start_dilation = self.list_of_dilations[0]
            if start_dilation == 1:
                out_weights = tf.Variable(tf.random_normal(shape=[self.hidden_unit_size, class_num]))
                out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
                fuse_outputs = outputs[-1]
            else:
                out_weights = tf.Variable(tf.random_normal(shape=[self.hidden_unit_size*start_dilation, class_num]))
                out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
                fuse_outputs = outputs[-start_dilation]
                for i in range(-start_dilation+1, 0, 1):
                    fuse_outputs = tf.concat([fuse_outputs, outputs[i]], axis = 1)
            
            log_predictions = tf.add(tf.matmul(fuse_outputs, out_weights), out_bias)
                
        elif experiment == "copy_memory" or experiment == "PTB":
            
            out_weights = tf.Variable(tf.random_normal(shape=[self.hidden_unit_size, class_num]))
            out_bias = tf.Variable(tf.random_normal(shape=[class_num]))
            
            outputs = tf.stack(outputs, axis = 0)
            out_h = tf.einsum('ijk,kl->jil', outputs, out_weights)
            log_predictions = tf.add(out_h, out_bias)
            
        else:
            
            print("Wrong selection for the variable 'experiment'")
            
        return log_predictions