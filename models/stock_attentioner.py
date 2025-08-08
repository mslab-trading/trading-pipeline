import torch
import torch.nn as nn

class StockAttention(nn.Module):  
    """
    Applies self-attention along the stock dimension.
    Input/Output shape: (batch_size, num_stocks, hidden_dim)
    """
    def __init__(self, num_stocks, hidden_dim, num_heads=4, dropout=0.1):
        """
        Args:
            num_stocks (int): Number of stocks (treated as sequence length).
            hidden_dim (int): Dimensionality of each token embedding.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention/output.
        """
        super(StockAttention, self).__init__()
        
        # MultiheadAttention in PyTorch expects the shape (batch_size, seq_length, embed_dim)
        # if batch_first=True. Here, seq_length = num_stocks, embed_dim = hidden_dim.
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # A LayerNorm often helps stabilize training when using attention
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, num_stocks, hidden_dim)
        
        Returns:
            Tensor of shape (batch_size, num_stocks, hidden_dim)
        """
        # MultiHeadAttention (when batch_first=True) expects (batch_size, seq_length, embed_dim)
        # We do self-attention, so query, key, and value are all x.
        attn_output, attn_weights = self.mha(x, x, x)
        
        # Residual connection + Layer Normalization
        x = x + attn_output
        x = self.norm(x)
        
        return x

class LinearDecoder(nn.Module):
    """Decodes the embeddings into a single value per stock."""
    def __init__(self, hidden_dim):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return self.decoder(x).squeeze(-1)

class MovingAvg(nn.Module):
    """
    Moving average along the sequence-length (last-but-one) axis.
    Expects input of shape (B⋅N, F, L) after a view/permute from the caller.
    """
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.pool = nn.AvgPool1d(kernel_size=kernel_size,
                                 stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B⋅N, F, L)
        if self.padding:                           # replicate-padding on both ends
            left  = x[:, :, :1].expand(-1, -1, self.padding)
            right = x[:, :, -1:].expand(-1, -1, self.padding)
            x = torch.cat([left, x, right], dim=2) # (B⋅N, F, L + 2·pad)
        return self.pool(x)                        # (B⋅N, F, L)
        

class SeriesDecomp(nn.Module):
    """
    Decomposes a series into (residual, trend) with a moving average
    along the sequence-length axis.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.mavg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        """
        x : (B, N, L, F)
        returns:
            res   : (B, N, L, F)  # x minus moving-average
            trend : (B, N, L, F)  # moving-average component
        """
        B, N, L, F = x.shape

        # --- move to (B·N, F, L) for AvgPool1d ---
        x_ = x.permute(0, 1, 3, 2).reshape(B * N, F, L)

        trend_ = self.mavg(x_)                    # (B·N, F, L)

        # --- reshape back to original 4-D layout ---
        trend = trend_.view(B, N, F, L).permute(0, 1, 3, 2)

        res = x - trend
        return res, trend

class LSTMEncoder(nn.Module):
    """Encodes the time series data for each stock into a fixed-length embedding."""
    def __init__(self, feature_dim, seq_len, hidden_dim, use_fft, num_layers=1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.use_fft = use_fft
        self.feature_mixer = nn.Linear(feature_dim * 2, feature_dim) if self.use_fft else nn.Linear(feature_dim, feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.encoder = iTransformer_encoder(seq_len, hidden_dim)

    def forward(self, x):
        # feature mixer
        x = x.contiguous() 
        batch_size, num_stocks, seq_len, feature_dim = x.shape
        x = x.view(batch_size * num_stocks * seq_len, feature_dim)

        if self.use_fft:
            # Apply FFT along the last dimension (time dimension)
            x = x.view(batch_size * num_stocks, seq_len, feature_dim)
            x = torch.fft.fft(x, dim=-1)
            x1 = x.real
            x2 = x.imag
            x = torch.cat([x1, x2], dim=-1)  # Concatenate real and imaginary parts
            x = x.view(batch_size * num_stocks, seq_len, feature_dim * 2)

        # feature mixing
        x = self.feature_mixer(x)
        x = x.view(batch_size, num_stocks, seq_len, feature_dim)
        
        # LSTM encoder
        x = x.view(batch_size * num_stocks, seq_len, feature_dim).contiguous()
        _, (hidden_state, _) = self.lstm(x)
        hidden_state = hidden_state[-1]  # Take last layer's hidden state
        hidden_state = hidden_state.view(batch_size, num_stocks, -1)

        return hidden_state

class StockEncoderLSTM(nn.Module):
    """Encodes the stock data into a fixed-length embedding."""
    def __init__(self, feature_dim, seq_len, hidden_dim, use_fft, num_layers=1, bidirectional=False):
        super(StockEncoderLSTM, self).__init__()
        
        self.decomposition = SeriesDecomp()
        
        self.seasonal_model = LSTMEncoder(
            feature_dim=feature_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            use_fft=use_fft,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        self.trend_model = LSTMEncoder(
            feature_dim=feature_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            use_fft=use_fft,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

    def forward(self, x):
        seasonal_x, trend_x = self.decomposition(x)
        seasonal_emb = self.seasonal_model(seasonal_x)  # (batch_size, num_stocks, hidden_dim)
        trend_emb = self.trend_model(trend_x)
        return seasonal_emb + trend_emb  # (batch_size, num_stocks, hidden_dim)
    
class BrokerEncoderLSTM(nn.Module):
    """Encodes the broker data into a fixed-length embedding."""
    def __init__(self, feature_dim, seq_len, hidden_dim, num_layers=1, bidirectional=False):
        super(BrokerEncoderLSTM, self).__init__()
        self.feature_mixer = nn.Linear(feature_dim, feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.encoder = iTransformer_encoder(seq_len, hidden_dim)

    def forward(self, x):
        # feature mixer
        batch_size, num_stocks, seq_len, feature_dim = x.shape
        x = x.view(batch_size * num_stocks * seq_len, feature_dim)
        x = self.feature_mixer(x)
        x = x.view(batch_size, num_stocks, seq_len, feature_dim)
        
        # LSTM encoder
        x = x.view(batch_size * num_stocks, seq_len, feature_dim)
        _, (hidden_state, _) = self.lstm(x)
        hidden_state = hidden_state[-1]  # Take last layer's hidden state
        hidden_state = hidden_state.view(batch_size, num_stocks, -1)

        return hidden_state
    
class GeneralDataEncoderLSTM(nn.Module):
    """Encodes the broker data into a fixed-length embedding."""
    def __init__(self, feature_dim, seq_len, hidden_dim, num_layers=1, bidirectional=False):
        super(GeneralDataEncoderLSTM, self).__init__()
        self.feature_mixer = nn.Linear(feature_dim, feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.encoder = iTransformer_encoder(seq_len, hidden_dim)

    def forward(self, x):
        # feature mixer
        batch_size, seq_len, feature_dim = x.shape
        x = self.feature_mixer(x)
        
        # LSTM encoder
        _, (hidden_state, _) = self.lstm(x)
        hidden_state = hidden_state[-1]  # Take last layer's hidden state
        hidden_state = hidden_state.view(batch_size, -1)

        return hidden_state
    

class StockAttentioner(nn.Module):
    """End-to-end model combining feature mixing, LSTM encoding, stock mixing, and decoding."""
    def __init__(self, args):
        self.broker = args.broker
        self.general_data = args.general_data
        
        super(StockAttentioner, self).__init__()
        
        self.stock_encoder = StockEncoderLSTM(
            feature_dim=args.feature_dim,
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            use_fft=args.use_fft,
            num_layers=1,
            bidirectional=False
        )
        
        if self.broker:
            self.broker_encoder = BrokerEncoderLSTM(
                feature_dim=args.broker_dim,
                seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
            )
            self.stock_broker_mixer = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        
        if self.general_data:
            self.general_encoder = GeneralDataEncoderLSTM(
                feature_dim=args.general_dim,
                seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
            )
        
        self.stock_attention = StockAttention(num_stocks=args.num_stocks, hidden_dim=args.hidden_dim)
        self.decoder = LinearDecoder(hidden_dim=args.hidden_dim)

    def forward(self, x, x_broker, x_general):
        
        if not self.broker:
            x = self.stock_encoder(x)
            x = self.stock_attention(x)
            y = self.decoder(x)
            return y
        
        # x: (batch_size, num_stocks, seq_len, feature_dim)
        # broker: (batch_size, num_stocks, seq_len, feature_dim)
        # breakpoint()
        x = self.stock_encoder(x)               # shape: (batch_size, num_stocks, hidden_dim)
        x_broker = self.broker_encoder(x_broker)    # shape: (batch_size, num_stocks, hidden_dim)
        
        x = torch.cat([x, x_broker], dim=-1)      # (batch_size, num_stocks, 2*hidden_dim)
        x = self.stock_broker_mixer(x)          # (batch_size, num_stocks, hidden_dim)
        
        if self.general_data:
            x_general = self.general_encoder(x_general)
            g = x_general.unsqueeze(1)
            new_x = torch.cat([x, g], dim=1)
            attended = self.stock_attention(new_x)   
            x = attended[:, : x.size(1), :] 
        else:
            x = self.stock_attention(x)             # (batch_size, num_stocks, hidden_dim)

        y = self.decoder(x)    
        return y