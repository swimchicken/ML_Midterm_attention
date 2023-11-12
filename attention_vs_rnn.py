import torch
import torch.nn as nn
import torch.cuda as cuda
import time


sequence_length = 100
input_size = 10
input_data = torch.randn((1, sequence_length, input_size))

input_data_gpu = input_data.cuda()


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class SimpleSelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SimpleSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads=1)

    def forward(self, x):
        out, _ = self.attention(x, x, x)
        return out


hidden_size = 20
rnn_model = SimpleRNN(input_size, hidden_size).cuda()
attention_model = SimpleSelfAttention(input_size).cuda()

start_event = cuda.Event(enable_timing=True)
end_event = cuda.Event(enable_timing=True)

start_event.record()
output_rnn = rnn_model(input_data_gpu)
end_event.record()
cuda.synchronize()
rnn_time = start_event.elapsed_time(end_event)

start_event.record()
output_attention = attention_model(input_data_gpu)
end_event.record()
cuda.synchronize()
attention_time = start_event.elapsed_time(end_event)

print(f"RNN 執行時間：{rnn_time / 1000} 秒")
print(f"Self-Attention 執行時間：{attention_time / 1000} 秒")
