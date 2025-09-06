# archs.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

# ---------- small utility ----------
def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize tensor x to match ref's HxW if they differ (bilinear, no align_corners)."""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    return x

# ---------- PGAM / GSConv blocks (lightweight) ----------
class PGAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_gate(x) * self.spatial_gate(x)

class GSConv(nn.Module):
    """Grouped-separable-ish conv block."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = max(c2 // 2, 1)
        p = k // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, k, s, p, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return torch.cat((x1, x2), 1)

class D_DoubleConv(nn.Module):
    """Decoder conv block with PGAM enrichment and residual (fixed)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = GSConv(in_ch, in_ch)
        self.conv2 = GSConv(in_ch, in_ch, k=5)
        self.conv3 = GSConv(in_ch, in_ch, k=7)

        self.pgam1 = PGAM(in_ch)
        self.pgam2 = PGAM(in_ch)
        self.pgam3 = PGAM(in_ch)
        self.pgam4 = PGAM(in_ch * 3)

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU6(inplace=True)

        # reduce (in_ch*3) -> out_ch
        self.conv4 = nn.Conv2d(in_ch * 3, out_ch, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch)

        # residual projects original input (in_ch) -> out_ch
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.out_act = nn.ReLU6(inplace=True)

    def forward(self, x):
        r = x  # save original input for residual

        x1 = self.act(self.bn1(self.conv1(self.pgam1(x))))
        x2 = self.act(self.bn2(self.conv2(self.pgam2(x))))
        x3 = self.act(self.bn3(self.conv3(self.pgam3(x))))

        z = torch.cat((x1, x2, x3), dim=1)  # (in_ch*3)
        z = self.pgam4(z)

        y = self.bn4(self.conv4(z))         # (out_ch)
        y = y + self.residual(r)            # add projected residual
        y = self.out_act(y)
        return y


# ---------- ASPP (multi-scale context) ----------
class ASPP(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 64):
        super().__init__()
        self.b1    = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.b3r6  = nn.Conv2d(in_ch, mid_ch, 3, padding=6, dilation=6, bias=False)
        self.b3r12 = nn.Conv2d(in_ch, mid_ch, 3, padding=12, dilation=12, bias=False)
        self.b3r18 = nn.Conv2d(in_ch, mid_ch, 3, padding=18, dilation=18, bias=False)
        self.proj  = nn.Conv2d(mid_ch * 4, in_ch, 1, bias=False)
        self.bn    = nn.BatchNorm2d(in_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b3r6(x), self.b3r12(x), self.b3r18(x)], dim=1)
        return self.act(self.bn(self.proj(y)))

# ---------- SCSE (skip gating) ----------
class SCSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)

# ---------- Light token-MLP block for C3 ----------
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.pw = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

    def forward(self, x, H, W):
        return self.pw(self.dw(x)) + x

class Lo2(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        self.act = act_layer()
        self.dwconv = DWConv(in_features)

    def forward(self, x, H, W):
        # x: (B, N, C)
        x1 = self.fc2(self.drop(self.act(self.fc1(x))))  # MLP
        x2 = x.transpose(1, 2).view(x.size(0), x.size(2), H, W)
        x2 = self.dwconv(x2, H, W).flatten(2).transpose(1, 2)
        return x1 + x2

class Lo2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = Lo2(dim, hidden_features=dim)

    def forward(self, x, H, W):
        return self.mlp(x, H, W)

# ---------- Simple Up block ----------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))

# ---------- Main model: EfficientNet-B3 encoder + MobileViT-S bottleneck + UNet decoder ----------
class EffB3_MobileViT_UNet(nn.Module):
    """
    - Encoder: timm efficientnet_b3 (features_only, out_indices=(1,2,3,4))
    - Bottleneck: timm mobilevit_s (features_only, last stage), projected to match C4
    - Decoder: UNet-style with ASPP at bottleneck, SCSE on skips, light token-MLP on C3
    - Output: 1xHxW probability map (sigmoid), same HxW as input
    """
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        import timm

        # Encoder
        self.encoder = timm.create_model(
            'efficientnet_b3', pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4), in_chans=input_channels
        )
        enc_chs = self.encoder.feature_info.channels()   # e.g. [24, 40, 112, 160]
        c1ch, c2ch, c3ch, c4ch = enc_chs

        # MobileViT bottleneck (timm, features_only)
        self.mvit = timm.create_model(
            'mobilevit_s', pretrained=True, features_only=True,
            out_indices=(4,), in_chans=input_channels
        )
        mvit_ch = self.mvit.feature_info.channels()[0]   # typically 640

        # Lateral projections to fixed dims
        self.lproj1 = nn.Conv2d(c1ch, 64, 1, bias=False)
        self.lproj2 = nn.Conv2d(c2ch, 128, 1, bias=False)
        self.lproj3 = nn.Conv2d(c3ch, 256, 1, bias=False)
        self.lproj4 = nn.Conv2d(c4ch, 320, 1, bias=False)

        # Project MobileViT to 320 and fuse with C4
        self.mvit_proj = nn.Conv2d(mvit_ch, 320, 1, bias=False)

        # ASPP on fused bottleneck
        self.aspp = ASPP(320, mid_ch=64)

        # Skip gating + attention on C3
        self.scse1 = SCSE(64)
        self.scse2 = SCSE(128)
        self.scse3 = SCSE(256)
        self.c3_attn = Lo2Block(dim=256)

        # Decoder
        self.up4 = Up(320, 256)               # 1/32 -> 1/16
        self.dec3 = D_DoubleConv(256 + 256, 256)

        self.up3 = Up(256, 128)               # 1/16 -> 1/8
        self.dec2 = D_DoubleConv(128 + 128, 128)

        self.up2 = Up(128, 64)                # 1/8  -> 1/4
        self.dec1 = D_DoubleConv(64 + 64, 64)

        self.up1 = Up(64, 32)                 # 1/4  -> 1/2
        self.dec0 = D_DoubleConv(32, 32)

        self.final = nn.Conv2d(32, num_classes, 1)

        # Optional deep heads (kept internal; we still return only the final prob map)
        if self.deep_supervision:
            self.seg_deep3 = nn.Conv2d(256, num_classes, 1)
            self.seg_deep2 = nn.Conv2d(128, num_classes, 1)
            self.seg_deep1 = nn.Conv2d(64,  num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]

        # ----- Encoder -----
        c1, c2, c3, c4 = self.encoder(x)      # strides ~ [4, 8, 16, 32]
        c1 = self.lproj1(c1)                  # 64
        c2 = self.lproj2(c2)                  # 128
        c3 = self.lproj3(c3)                  # 256
        c4 = self.lproj4(c4)                  # 320

        # Light attention on C3 (token-MLP)
        B, C, H3, W3 = c3.shape
        t3 = c3.flatten(2).transpose(1, 2)    # (B, N, C)
        t3 = self.c3_attn(t3, H3, W3)
        c3 = t3.transpose(1, 2).reshape(B, C, H3, W3)

        # SCSE gates on skips
        c3 = self.scse3(c3)
        c2 = self.scse2(c2)
        c1 = self.scse1(c1)

        # ----- Bottleneck (MobileViT-S) -----
        mv = self.mvit(x)[0]                  # last stage feature map
        mv = _resize_to(mv, c4)               # match spatial size to C4
        mv = self.mvit_proj(mv)               # -> 320
        bottleneck = self.aspp(c4 + mv)       # enrich with ASPP

        # ----- Decoder -----
        d3 = self.up4(bottleneck)             # -> size of c3
        d3 = _resize_to(d3, c3)
        d3 = self.dec3(torch.cat([d3, c3], dim=1))   # 256

        d2 = self.up3(d3)                     # -> size of c2
        d2 = _resize_to(d2, c2)
        d2 = self.dec2(torch.cat([d2, c2], dim=1))   # 128

        d1 = self.up2(d2)                     # -> size of c1
        d1 = _resize_to(d1, c1)
        d1 = self.dec1(torch.cat([d1, c1], dim=1))   # 64

        d0 = self.up1(d1)                     # -> 1/2
        d0 = self.dec0(d0)                    # 32

        # logits at 1/2, then upsample logits to full res and sigmoid
        logits = self.final(d0)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        probs  = torch.sigmoid(logits)
        return probs

# Keep your old import path working
HybridEffB3ViTUNet = EffB3_MobileViT_UNet

__all__ = ['EffB3_MobileViT_UNet', 'HybridEffB3ViTUNet']
