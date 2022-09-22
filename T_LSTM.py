#Here is the basic strcuture of T_LSTM based on paper 'Patient Subtyping cia Time-Aware LSTM Networks'
import torch
import torch.nn as nn
class T_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(T_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim, hidden_dim)

    def get_time_interval(self, time):
        '''
        convert time point to time interval
        '''
        time_interval = torch.zeros(time.shape[0], time.shape[1] - 1)
        for col in range(time.shape[1] - 1):
            time_interval[:, col] = time[:, col + 1] - time[:, col]
        return time_interval


    def time_interval_weight(self, time_interval):
        '''
        calculate weight of time interval
        '''
        c1 = torch.tensor(1, dtype = torch.float32)
        c2 = torch.tensor(2.7193, dtype = torch.float32)
        time_weight = torch.div(c1, torch.log(time_interval + c2))
        return time_weight


    def forward(self, inputs, time):
        batch_size, seq_len, input_size = inputs.size()
        time_interval = self.get_time_interval(time)

        #initial hidden state and cell state
        h = torch.zeros(batch_size, self.hidden_dim)
        c = torch.zeros(batch_size, self.hidden_dim)

        for i in range(seq_len - 1):
            #In this part, we need to ignore the effect of the last folloewup
            #follow the steps shown in paper
            C_t_1_S = torch.tanh(self.linear_hidden(c)) #short-term memory
            C_t_1_discount = C_t_1_S * self.time_interval_weight(time_interval[:, i]).unsqueeze(1).repeat(1, C_t_1_S.shape[1]) #discounted short-term memory
            C_t_1_T = c - C_t_1_S #Long-term memory
            C_t_1_star = C_t_1_T + C_t_1_discount #Adjusted previous memory
            f_t = torch.sigmoid(self.linear_input(inputs[:, i]) + self.linear_hidden(h))
            i_t = torch.sigmoid(self.linear_input(inputs[:, i]) + self.linear_hidden(h))
            o_t = torch.sigmoid(self.linear_input(inputs[:, i]) + self.linear_hidden(h))
            C_bar = torch.tanh(self.linear_input(inputs[:, i]) + self.linear_hidden(h))
            c = f_t * C_t_1_star + i_t * C_bar
            h = o_t * torch.tanh(C_bar)

        res = self.linear_hidden(o_t)
        outputs = torch.stack((res, h, c))

        return outputs

def test():
    patient = torch.randn(128, 4, 10)
    time = torch.randn(128, 4)
    model = T_LSTM(patient.shape[2], 20)
    outputs = model(patient, time)
    print(outputs.shape)

if __name__ == '__main__':
    test()
