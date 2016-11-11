require 'nn'
require 'nngraph'

local LSTM = {}
function LSTM.lstm(input_size_v, input_size_w, output_size, rnn_size, dropout)
    dropout = dropout or 0 

    -- there will be 4 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- visual embedding
    table.insert(inputs, nn.Identity()()) -- word embedding
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]

    -- c, h from previous timestep
    local xv = inputs[1]  -- visual embedding
    local xw = inputs[2]  -- word embedding
    local prev_c = inputs[3]
    local prev_h = inputs[4]

    -- the input to this layer
    if dropout > 0 then xw = nn.Dropout(dropout)(xw) end  --apply dropout on word-embedding

    -- evaluate the input sums at once for efficiency
    local v2h = nn.Linear(input_size_v, 4 * rnn_size)(xv)
    local w2h = nn.Linear(input_size_w, 4 * rnn_size)(xw)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({nn.CAddTable()({v2h, w2h}), h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
                        nn.CMulTable()({forget_gate, prev_c}),
                        nn.CMulTable()({in_gate,     in_transform})
                    })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    -- set up the decoder
    local top_h = next_h
    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
    local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
    local logsoft = nn.LogSoftMax()(proj)

    -- here's the outputs, and return
    local outputs = {next_c, next_h, logsoft}
    return nn.gModule(inputs, outputs)
end

return LSTM
