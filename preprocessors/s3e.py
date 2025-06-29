import torch
import torch.nn as nn

from models.stock_attentioner import SeriesDecomp

class BaseLSTM(nn.Module):
    def __init__(self, args):
        super(BaseLSTM, self).__init__()
        self.feature_mixer = nn.Linear(args.feature_dim, args.feature_dim)
        self.lstm = nn.LSTM(
            input_size=args.feature_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.linear = nn.Linear(args.hidden_dim, args.feature_dim)
    
    def forward(self, x):
        # feature mixer
        x = x.contiguous()
        batch_size, num_stocks, seq_len, feature_dim = x.shape
        x = x.view(batch_size * num_stocks * seq_len, feature_dim)
        x = self.feature_mixer(x)
        x = x.view(batch_size, num_stocks, seq_len, feature_dim)
        
        # LSTM encoder
        x = x.contiguous()
        x = x.view(batch_size * num_stocks, seq_len, feature_dim)
        output, (hn, cn) = self.lstm(x)
        output = output.view(batch_size, num_stocks, seq_len, -1)
        output = self.linear(output)

        return output

class BaseTransformer(nn.Module):
    def __init__(self, args):
        super(BaseTransformer, self).__init__()
        self.d_model = args.d_model
        self.feature_mixer = nn.Linear(args.feature_dim, self.d_model)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=args.nhead,
                dim_feedforward=args.hidden_dim,
                dropout=args.dropout
            ),
            num_layers=args.num_layers
        )
        self.linear = nn.Linear(self.d_model, args.feature_dim)
    
    def forward(self, x):
        # feature mixer
        x = x.contiguous()
        batch_size, num_stocks, seq_len, feature_dim = x.shape
        x = x.view(batch_size * num_stocks * seq_len, feature_dim)
        x = self.feature_mixer(x)
        x = x.view(batch_size, num_stocks, seq_len, self.d_model)

        # Transformer encoder
        x = x.contiguous()
        x = x.view(batch_size * num_stocks, seq_len, self.d_model)
        output = self.model(x)
        output = output.view(batch_size, num_stocks, seq_len, self.d_model)
        output = self.linear(output)
        return output

class S3E(nn.Module):
    def __init__(self, args):
        super(S3E, self).__init__()
        self.args = args
        self.feature_dim = args.feature_dim

        self.decomposition = SeriesDecomp()

        model_mapping = {
            'LSTM': BaseLSTM,
            'Transformer': BaseTransformer
        }

        self.seasonal_model = model_mapping[args.base_model](args)
        self.trend_model = model_mapping[args.base_model](args)

        self.projector = nn.Linear(
            args.feature_dim * args.seq_len * 2, 
            1
        )

    def forward(self, x):
        seasonal_x, trend_x = self.decomposition(x)
        seasonal_emb = self.seasonal_model(seasonal_x)  # (batch_size, num_stocks, seq_len, hidden_dim)
        trend_emb = self.trend_model(trend_x)
        return seasonal_emb + trend_emb

    def forward_train(self, Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general):
        x = self.forward(Xs).flatten(-2)  # (batch_size, num_stocks, seq_len * hidden_dim)
        x2 = self.forward(X2s).flatten(-2)  # (batch_size, num_stocks, seq_len * hidden_dim)
        x = torch.cat((x, x2), dim=-1)
        output = self.projector(x).squeeze(-1) # (batch_size, num_stocks)
        return output
