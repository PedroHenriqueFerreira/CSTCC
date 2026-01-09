import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolução => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Desce a resolução da imagem: MaxPool => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Sobe a resolução e concatena: Upsample => Concat => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample bilinear é leve e não tem parâmetros aprendíveis
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Input vindo da camada anterior (fundo da rede)
        x2: Input vindo da Skip Connection (encoder)
        """
        x1 = self.up(x1)
        
        # O torch.cat junta os canais. x2 deve ter o mesmo tamanho espacial de x1
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ColorCorrectionModel(nn.Module):
    def __init__(self, n_conditions=4, base=32):
        super().__init__()
        
        # --- Configurações ---
        embed_dim = 4
        self.embedding = nn.Embedding(n_conditions, embed_dim)
        
        # A entrada é RGB (3) + Canais de Embedding (4)
        c_in = 3 + embed_dim 
        
        # --- Arquitetura (Definição dos Canais) ---
        # Encoder
        self.inc   = DoubleConv(c_in, base)       # 32
        
        self.down1 = Down(base,     base*2)       # 64
        self.down2 = Down(base*2,   base*4)       # 128
        self.down3 = Down(base*4,   base*8)       # 256 (Bottleneck)
        
        # Decoder
        # No Up, o 'in_channels' é a soma do que vem de baixo + o skip
        # Ex: up1 recebe 256 (bottleneck) + 128 (skip do down2) = 384 canais de entrada
        self.up1 = Up(base*8 + base*4, base*4)    # Entra 384, Sai 128
        self.up2 = Up(base*4 + base*2, base*2)    # Entra 192, Sai 64
        self.up3 = Up(base*2 + base,   base)      # Entra 96,  Sai 32

        # Saída
        self.outc = nn.Conv2d(base, 3, kernel_size=1)
        
        self.final_activation = nn.Sigmoid()

    def forward(self, x, condition_idx):
        B, C, H, W = x.shape
        
        # 1. Guardar a referência da imagem original para a soma final
        input_img = x 
        
        # 2. Preparar Condição
        c = self.embedding(condition_idx)
        # Expande (B, embed) -> (B, embed, H, W)
        c_expanded = c.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # Concatena na entrada
        x_in = torch.cat([input_img, c_expanded], dim=1)

        # 3. Encoder
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Bottleneck

        # 4. Decoder (com Skip Connections)
        # Aqui 'x' passa a ser o resultado das convoluções, e não mais a imagem original
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1) # Neste ponto, x tem 32 canais (base)

        # 5. Saída Residual
        # A conv1x1 transforma os 32 canais em 3 canais (RGB Delta)
        delta = self.outc(x)
        delta = self.final_activation(delta)
        
        # CORREÇÃO: Somar delta com 'input_img' (original), não com 'x' (features)
        # return torch.clamp(input_img + delta, 0.0, 1.0)
        
        # return torch.clamp(delta, 0.0, 1.0)
        return delta