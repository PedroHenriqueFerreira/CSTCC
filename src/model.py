from torch import cat, nn, Tensor

class DoubleConv(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int, kernel_size = 3): 
        super().__init__() 
        self.double_conv = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels, affine=True), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels, affine=True), 
            nn.ReLU(inplace=True) 
        ) 
    
    def forward(self, x: Tensor) -> Tensor: 
        return self.double_conv(x) 
    
class Down(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int): 
        super().__init__() 
        self.down_conv = nn.Sequential( 
            nn.MaxPool2d(2), 
            DoubleConv(in_channels, out_channels) 
        ) 
    
    def forward(self, x: Tensor) -> Tensor: 
        return self.down_conv(x) 
    
class Up(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int): 
        super().__init__() 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear') 
        self.conv = DoubleConv(in_channels, out_channels) 
    
    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor: 
        return self.conv(cat([x_enc, self.up(x_dec)], dim=1)) 

class ColorCorrectionModel(nn.Module): 
    def __init__(self, conditions = 4, embedding_dim = 16, base = 32): 
        super().__init__() 
        self.embedding = nn.Embedding(conditions, embedding_dim) 
        
        self.inc = DoubleConv(3 + embedding_dim, base) 
        
        self.down1 = Down(base, base * 2) 
        self.down2 = Down(base * 2, base * 4) 
        self.down3 = Down(base * 4, base * 8) 
        
        self.up1 = Up(base * 8 + base * 4, base * 4) 
        self.up2 = Up(base * 4 + base * 2, base * 2) 
        self.up3 = Up(base * 2 + base, base) 
        
        self.outc = nn.Conv2d(base, 3, kernel_size=1) 
        
    def forward(self, x: Tensor, condition: Tensor) -> Tensor: 
        B, C, H, W = x.shape 
        c = self.embedding(condition).view(B, -1, 1, 1).expand(-1, -1, H, W) 
        x = cat([x, c], dim=1) 
        
        x1 = self.inc(x) 
        
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x = self.down3(x3) 
        
        x = self.up1(x, x3) 
        x = self.up2(x, x2) 
        x = self.up3(x, x1) 
        
        x = self.outc(x) 
        
        return x.sigmoid()