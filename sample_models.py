from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Activation, GaussianNoise,
                          TimeDistributed, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)


def simple_rnn_model(name, input_dim, output_dim=29):
    """ Build a simple recurrent network for speech recognition """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add recurrent layer
    rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(rnn)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    m.output_length = lambda x: x
    print(m.summary())
    return m


def rnn_model(name, input_dim, rnn_act, units, output_dim=29):
    """ Build a recurrent + dense network for speech recognition """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add recurrent layer
    rnn = GRU(units, activation=rnn_act, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # Add time distributed dense layer
    td_dense = TimeDistributed(Dense(output_dim, name='dense'), name='td_dense')(bn_rnn)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(td_dense)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    m.output_length = lambda x: x
    print(m.summary())
    return m


def cnn_rnn_model(name, input_dim, filters, kernel_size, conv_stride, conv_border_mode, conv_act, units, output_dim=29):
    """ Build a convolutional + recurrent network for speech recognition """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add convolutional layer
    cnn = Conv1D(filters, kernel_size,
                 strides=conv_stride,
                 padding=conv_border_mode,
                 activation=conv_act,
                 name='cnn')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_cnn')(cnn)
    # Add recurrent layer
    rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # Add time distributed dense layer
    td_dense = TimeDistributed(Dense(output_dim, name='dense'), name='td_dense')(bn_rnn)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(td_dense)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    m.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(m.summary())
    return m


def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    else:
        raise Exception(f'ERROR! Unsupported convolution border mode: {border_mode}')
    return (output_length + stride - 1) // stride


def deep_rnn_model(name, input_dim, rnn_layers, units, output_dim=29):
    """ Build a deep recurrent network for speech recognition """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add recurrent layers, each with batch normalization
    hld = input_data
    for layer in range(rnn_layers):
        rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name=f'rnn_{layer+1}')(hld)
        hld = BatchNormalization(name=f'bn_rnn_{layer+1}')(rnn)
    # Add time distributed dense layer
    td_dense = TimeDistributed(Dense(output_dim, name='dense'), name='td_dense')(hld)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(td_dense)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    m.output_length = lambda x: x
    print(m.summary())
    return m


def bidirectional_rnn_model(name, input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech recognition """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add bidirectional recurrent layer
    rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')
    bd_rnn = Bidirectional(rnn, name='bd_rnn')(input_data)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(bd_rnn)
    # Add time distributed dense layer
    td_dense = TimeDistributed(Dense(output_dim, name='dense'), name='td_dense')(bn_rnn)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(td_dense)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    m.output_length = lambda x: x
    print(m.summary())
    return m


def tail_rnn_model(name, input_dim, dense_layers, dense_act, dense_units, units, output_dim=29):
    """ Build a recurrent + deep dense network for speech recognition """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add recurrent layer
    rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization
    hld = BatchNormalization(name='bn_rnn')(rnn)
    # Add time distributed dense layers, each with batch normalization and dropout
    for layer in range(dense_layers - 1):
        dense = Dense(dense_units, activation=dense_act, name=f'dense_{layer+1}')
        td_dense = TimeDistributed(dense, name=f'td_dense_{layer+1}')(hld)
        dp_dense = Dropout(0.5, name=f'dp_dense_{layer+1}')(td_dense)
        hld = BatchNormalization(name=f'bn_dense_{layer+1}')(dp_dense)
    td_dense = TimeDistributed(Dense(output_dim, name=f'dense_{dense_layers}'), name=f'td_dense_{dense_layers}')(hld)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(td_dense)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    m.output_length = lambda x: x
    print(m.summary())
    return m


def final_model(name='model_end', input_dim=13, noise=0.075,
                filters=256, kernel_size=5,
                rnn_type=2, rnn_bd=True, rnn_layers=3, units=256,
                dense_layers=2, dense_units=256, output_dim=29):
    """ Build a deep convolutional + deep recurrent + deep dense network for speech recognition """
    # Define recurrent layers
    rnn_cls = [SimpleRNN, LSTM, GRU]
    assert rnn_type in range(len(rnn_cls))
    # Main acoustic input
    input_data = Input(shape=(None, input_dim), name='the_input')
    # Add Gaussian noise with batch normalization
    noise_data = GaussianNoise(noise, name='noise_input')(input_data) if noise > 0.0 else input_data
    bn_data = BatchNormalization(name='bn_input')(noise_data)
    # Add convolutional layers, each with batch normalization and dropout
    cnn_1 = Conv1D(filters, kernel_size, strides=1, padding='same', activation='relu', name='cnn_1')(bn_data)
    dp_cnn_1 = Dropout(0.25, name='dp_cnn_1')(cnn_1)
    bn_cnn_1 = BatchNormalization(name='bn_cnn_1')(dp_cnn_1)
    cnn_2 = Conv1D(filters, kernel_size, strides=2, padding='valid', activation='relu', name='cnn_2')(bn_cnn_1)
    dp_cnn_2 = Dropout(0.25, name='dp_cnn_2')(cnn_2)
    bn_cnn_2 = BatchNormalization(name='bn_cnn_2')(dp_cnn_2)
    # Add bidirectional recurrent layers, each with batch normalization and dropout
    hld = bn_cnn_2
    for layer in range(rnn_layers):
        rnn = rnn_cls[rnn_type](units, activation='relu', return_sequences=True,
                                implementation=2, dropout=0.25, name=f'rnn_{layer+1}')
        bd_rnn = Bidirectional(rnn, name=f'bd_rnn_{layer+1}')(hld) if rnn_bd else rnn(hld)
        hld = BatchNormalization(name=f'bn_rnn_{layer+1}')(bd_rnn)
    # Add time distributed dense layers, each with batch normalization and dropout
    for layer in range(dense_layers-1):
        dense = Dense(dense_units, activation='relu', name=f'dense_{layer+1}')
        td_dense = TimeDistributed(dense, name=f'td_dense_{layer+1}')(hld)
        dp_dense = Dropout(0.5, name=f'dp_dense_{layer+1}')(td_dense)
        hld = BatchNormalization(name=f'bn_dense_{layer+1}')(dp_dense)
    td_dense = TimeDistributed(Dense(output_dim, name=f'dense_{dense_layers}'), name=f'td_dense_{dense_layers}')(hld)
    # Add softmax activation layer
    output_data = Activation('softmax', name='softmax')(td_dense)
    # Specify the model
    m = Model(inputs=input_data, outputs=output_data, name=name)
    # Specify model.output_length
    m.output_length = lambda x: cnn_output_length(cnn_output_length(x, kernel_size, 'same', 1), kernel_size, 'valid', 2)
    print(m.summary())
    return m
