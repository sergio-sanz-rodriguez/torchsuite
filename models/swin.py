import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# =============================================================================
# Functions: 1) window_partition, 2) window_reverse
# Classes:   1) WindowAttention 2) SwinBlock 3) PatchMerging
#            4) PatchEmbed      5) BasicLayer (stage) 6) Vanilla SwinTransformer
# =============================================================================


def window_partition(x, window_size):

    """
    Partition an input feature map into non-overlapping windows.

    This function takes a 4D tensor in NHWC format and splits the spatial
    dimensions (H, W) into fixed-size windows of shape
    (window_size, window_size). Each window is returned as an independent
    tensor, stacked along the batch dimension.

    Description:
        Given an input tensor of shape (B, H, W, C), the output tensor has
        shape (B * num_windows, window_size, window_size, C), where
        num_windows = (H / window_size) * (W / window_size).

    Inputs:
        x (torch.Tensor):
            Input tensor of shape (B, H, W, C). The tensor must be contiguous
            in memory, and H and W must be divisible by window_size.
        window_size (int):
            The spatial size of each window.

    Returns:
        torch.Tensor:
            A tensor containing all windows, with shape
            (B * num_windows, window_size, window_size, C).
    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, window_size, window_size, C)  # -1 = B * (H//ws) * (W//ws)
    return x


def window_reverse(windows, win, H, W):

    """
    Reverse the window partition operation and reconstruct the feature map.

    This function takes a batch of windowed feature maps and rearranges them
    back into a single spatial feature map in NHWC format.

    Description:
        Given a tensor of windows with shape
        (B * num_windows, win, win, C), this function reconstructs the
        original feature map of shape (B, H, W, C), where:
            num_windows = (H / win) * (W / win)

    Inputs:
        windows (torch.Tensor):
            Input tensor containing windowed features with shape
            (B * num_windows, win, win, C).
        win (int):
            Spatial size of each window.
        H (int):
            Height of the reconstructed feature map.
        W (int):
            Width of the reconstructed feature map.

    Returns:
        torch.Tensor:
            Reconstructed feature map of shape (B, H, W, C).
    """

    B = int(windows.shape[0] / (H * W / win**2))
    x = windows.view(B, H // win, W // win, win, win, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    
    """
    Window-based multi-head self-attention (W-MSA) module with relative positional bias.

    Computes attention inside non-overlapping windows of the feature map.
    Adds learnable relative positional bias for each query-key pair inside the window.

    Attributes:
        dim (int): token feature dimension
        num_heads (int): number of attention heads
        head_dim (int): feature dimension per head
        scale (float): scaling factor for attention (1/sqrt(head_dim))
        win (int): window size (H=W=win)
        q, k, v (nn.Linear): linear projections for query, key, value
        proj (nn.Linear): output projection after attention
        pos_index (torch.Tensor): precomputed relative positional indices
        rel_bias (nn.Parameter): learnable relative positional bias table

    Note: numerical examples are given assuming a window size of 3x3.
    """

    def __init__(self, dim, num_heads, win):
        super().__init__()

        # ---------------- Basic parameters ----------------
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.win = win

        # ---------------- Linear projections ----------------
        # These project each token into Q, K, V spaces
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # ---------------- Relative positional bias ----------------
        # Create a coordinate grid for a window: shape (2, win, win)
        # Example win=3:
        # coords[0] = row coords: [[0,0,0],[1,1,1],[2,2,2]]
        # coords[1] = col coords: [[0,1,2],[0,1,2],[0,1,2]]
        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing='ij'))

        # Flatten to 2 x N (N=win*win)
        # Flatten to 2 x N (N=win*win)
        # Example win=3:
        # coords_flatten[0] = [0,0,0,1,1,1,2,2,2]
        # coords_flatten[1] = [0,1,2,0,1,2,0,1,2]
        coords_flatten = torch.flatten(coords, 1)  # shape: (2, 9)

        # Compute relative coordinates between all tokens in window
        # rel[:, i, j] = coords[:, i] - coords[:, j]        
        # Shape after this: (2, N, N) = (2, 9, 9)
        # [[[ 0,  0,  0, -1, -1, -1, -2, -2, -2],
        #   [ 0,  0,  0, -1, -1, -1, -2, -2, -2],
        #   ...],
        #  [[ 0, -1, -2,  0, -1, -2,  0, -1, -2],
        #   [ 0, -1, -2,  0, -1, -2,  0, -1, -2],
        #   ...]]
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)

        # Permute to shape (N, N, 2) for easier indexing: rel[q,k] = [drow, dcol]
        rel = rel.permute(1, 2, 0).contiguous()  # (N, N, 2)

        # Shift dx, dy so all values >=0
        # Each value ranges from 0 to (2*win-2) = 4 for win=3
        rel[:, :, 0] += (win - 1)
        rel[:, :, 1] += (win - 1)

        # Convert 2D relative coordinates to single integer index
        # index = drow * (2*win-1) + dcol
        rel[:, :, 0] *= (2 * win - 1)
        # Example 9x9 index matrix with unique values:
        # tensor([
        # [12,11,10, 7, 6, 5, 2, 1, 0],
        # [13,12,11, 8, 7, 6, 3, 2, 1],
        # [14,13,12, 9, 8, 7, 4, 3, 2],
        # [17,16,15,12,11,10, 7, 6, 5],
        # [18,17,16,13,12,11, 8, 7, 6],
        # [19,18,17,14,13,12, 9, 8, 7],
        # [22,21,20,17,16,15,12,11,10],
        # [23,22,21,18,17,16,13,12,11],
        # [24,23,22,19,18,17,14,13,12]
        # ])
        # Each element tells which of the 25 possible relative positions that query–key pair corresponds to.
        index = rel.sum(-1)  # shape: (N, N) = (9, 9)

        # Register as buffer (non-learnable, moves with .to(device))
        self.register_buffer('pos_index', index, persistent=False)

        # Learnable table of relative positional biases
        # Shape: ((2*win-1)*(2*win-1), num_heads)
        self.rel_bias = nn.Parameter(torch.zeros((2 * win - 1) * (2 * win - 1), num_heads))
        trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x, mask=None):
        
        """
        Forward pass.

        Args:
            x: (B_, N, C) = batch, tokens per window, token dim
               Example for win=3: N=9
            mask: optional attention mask for shifted windows (num_windows, N, N)

        Returns:
            out: (B_, N, C)
        """

        # Unpack shape
        B_, N, C = x.shape

        # Project to Q, K, V
        q = self.q(x)  # (B_, N, C)
        k = self.k(x)
        v = self.v(x)

        # Reshape for multi-head attention
        # (B_, N, C) -> (B_, N, num_heads, head_dim)
        q = q.view(B_, N, self.num_heads, self.head_dim)
        k = k.view(B_, N, self.num_heads, self.head_dim)
        v = v.view(B_, N, self.num_heads, self.head_dim)

        # Transpose to (B_, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention logits
        # (B_, heads, N, head_dim) @ (B_, heads, head_dim, N) -> (B_, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ---------------- Add relative positional bias ----------------
        # pos_index maps each query–key pair (i,j) to one of 25 unique relative positions.
        # self.rel_bias contains a learnable bias for each relative position and each attention head.
        # self.rel_bias[self.pos_index.view(-1)] “looks up” the correct bias for all 81 pairs at once.
        # .view(N, N, -1) reshapes it into a matrix form aligned with the attention scores.
        # .permute(2,0,1).unsqueeze(0) makes it ready to add to attn for all heads and batches.
        # At the end of the day, we are adding a learnable bias to each NxN query–key pair (attention score).
        # However, we don't need to learn a separate bias for all N×N pairs.
        # Instead, we only learn one bias for each of the (2*win-1)*(2*win-1) possible relative positions.
        # The pos_index tensor maps each query–key pair to one of these unique relative positions.

        # rel_bias shape: (25, num_heads)
        # pos_index shape: (N, N) = (9, 9)
        # Step 1: flatten index map: (81,)
        # Step 2: lookup rel_bias → (81, num_heads)
        # Step 3: reshape → (9, 9, num_heads)
        # Lookup: (N*N, heads) -> reshape to (N, N, heads)
        rel_bias = self.rel_bias[self.pos_index.view(-1)].view(N, N, -1)

        # Permute to (heads, N, N) and broadcast batch
        attn = attn + rel_bias.permute(2, 0, 1).unsqueeze(0)  # (B_, heads, N, N)

        # Apply mask if provided (for shifted windows), before softmax
        if mask is not None:
            num_win = mask.shape[0]          # number of windows per image
            B = B_ // num_win                # original batch size
            mask = mask.to(device=attn.device, dtype=attn.dtype)

            # Reshape to group windows per image:
            # (B_, heads, N, N) -> (B, num_win, heads, N, N)
            attn = attn.view(B, num_win, self.num_heads, N, N)

            # mask.unsqueeze(0).unsqueeze(2): (1, num_win, 1, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)

            # Back to (B_, heads, N, N)
            attn = attn.view(B_, self.num_heads, N, N)

        # Softmax over keys
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        out = attn @ v  # (B_, heads, N, head_dim)

        # Merge heads: (B_, heads, N, head_dim) -> (B_, N, C)
        out = out.transpose(1, 2).reshape(B_, N, C)

        # Output projection
        out = self.proj(out)
        return out


class SwinBlock(nn.Module):
    
    """
    Swin Transformer Block.

    This block implements either:
    - Window-based self-attention (shift = 0), or
    - Shifted window-based self-attention (shift = win // 2)

    It follows the standard Transformer structure:
    LayerNorm -> Window Attention -> Residual
    LayerNorm -> MLP -> Residual

    Attributes:
        dim (int): token embedding dimension (e.g. 96)
        res (Tuple[int, int]): spatial resolution (H, W), e.g. (9, 9)
        win (int): window size (e.g. 3)
        shift (int): window shift size (0 or win//2)
        mask (torch.Tensor or None): attention mask for shifted windows

    Note: numerical examples assume H = W = 9, win = 3, shift = 1.
    """

    def __init__(self, dim, res, win, shift, num_heads, mlp_ratio=4):
        super().__init__()

        """
        Store basic configuration.
        """
        self.dim = dim
        self.res = res
        self.win = win
        self.shift = shift
        self.num_heads = num_heads

        assert 0 <= self.shift < self.win, "shift must satisfy 0 <= shift < win"
        H, W = res
        assert H % win == 0 and W % win == 0, "H and W must be divisible by win (or pad before SwinBlock)"

        # First LayerNorm applied before window attention.
        # Normalizes over the channel dimension 'dim'.
        self.layer_norm1 = nn.LayerNorm(dim)

        # Window-based multi-head self-attention.
        # Attention is computed independently inside each win x win window.
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, win=win)

        # Second LayerNorm applied before the MLP.
        self.layer_norm2 = nn.LayerNorm(dim)

        # Feed-forward network (MLP) used in Transformer blocks.
        # Structure: dim -> mlp_ratio*dim -> dim
        hidden = int(dim * mlp_ratio)

        # IMPORTANT FIX: you had self.mpl but later called self.mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden),
            nn.GELU(),
            nn.Linear(in_features=hidden, out_features=dim)
        )

        # Attention mask is only needed when windows are shifted.
        # For shift = 0, windows do not overlap and no mask is required.
        # Mask only for shifted windows; register as buffer so it moves with .to(device)
        if self.shift > 0:
            mask = self.create_mask(H, W, self.shift)  # (nW, win*win, win*win)
            self.register_buffer("attn_mask", mask, persistent=False)
        else:
            self.attn_mask = None

    def create_mask(self, H, W, shift):
    
        """
        Create attention mask for SW-MSA (shifted windows).

        Intuition:
          After cyclic shifting, a single (shifted) window can contain tokens that originally
          belonged to different *unshifted* windows. We must prevent attention across those
          "logical boundaries." We do this by assigning region IDs and masking pairs with
          different IDs.

        Output:
          attn_mask: (num_windows, win*win, win*win)
            0.0      where attention is allowed (same region ID)
            -INF-ish where blocked (different region ID)

        Example parameters (used in comments below):
          H = W = 9, win = 3, shift = 1
        """

        # Build a "region id image"
        # Shape: (1, H, W, 1)
        img_mask = torch.zeros((1, H, W, 1), dtype=torch.int64)

        # Divide height and width into three slices each
        # With H=9, win=3, shift=1:
        #   slice(0, -win)      = rows 0..5
        #   slice(-win, -shift) = rows 6..7
        #   slice(-shift, None) = row 8
        h_slices = (slice(0, -self.win), slice(-self.win, -shift), slice(-shift, None))
        w_slices = (slice(0, -self.win), slice(-self.win, -shift), slice(-shift, None))

        # IMPORTANT FIX:
        # Your original code assigned region IDs twice per slice (and incremented cnt twice).
        # That makes region IDs wrong. We assign exactly once here.
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        """
        Conceptual region ID layout for H=W=9, win=3, shift=1 (before cyclic shift):

        0 0 0 0 0 0 | 1 1 | 2
        0 0 0 0 0 0 | 1 1 | 2
        0 0 0 0 0 0 | 1 1 | 2
        0 0 0 0 0 0 | 1 1 | 2
        0 0 0 0 0 0 | 1 1 | 2
        0 0 0 0 0 0 | 1 1 | 2
        ---------------------
        3 3 3 3 3 3 | 4 4 | 5
        3 3 3 3 3 3 | 4 4 | 5
        ---------------------
        6 6 6 6 6 6 | 7 7 | 8
        """

        # Partition img_mask into non-overlapping windows
        mask_windows = window_partition(img_mask, self.win)  # (num_windows, win, win, 1)

        """
        Windows (3x3) extracted from img_mask (H=W=9, win=3):

        Window 0 (rows 0..2, cols 0..2):
        0 0 0
        0 0 0
        0 0 0

        Window 1 (rows 0..2, cols 3..5):
        0 0 0
        0 0 0
        0 0 0

        Window 2 (rows 0..2, cols 6..8):
        1 1 2
        1 1 2
        1 1 2

        Window 6 (rows 6..8, cols 0..2):
        3 3 3
        3 3 3
        6 6 6

        Window 8 (rows 6..8, cols 6..8):
        4 4 5
        4 4 5
        7 7 8
        """

        # Flatten each window to (num_windows, win*win)
        mask_windows = mask_windows.view(-1, self.win * self.win)

        """
        Flatten examples (win=3 => win*win=9):

        Window 0:
        0 0 0
        0 0 0
        0 0 0 -> [0 0 0 0 0 0 0 0 0]

        Window 1 (top-middle) is also ALL 0 in this example:
        0 0 0
        0 0 0
        0 0 0 -> [0 0 0 0 0 0 0 0 0]

        Window 2 (top-right):
        1 1 2
        1 1 2
        1 1 2 -> [1 1 2 1 1 2 1 1 2]
        """

        # Compute pairwise differences to create attention mask
        # - attn_mask[w, j, k] = region_id[j] - region_id[k]
        # - 0   -> same region (attention allowed)
        # - !=0 -> different region (attention blocked)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        """
        Example for one flattened window (Window 2):

        IDs:
        [1, 1, 2,
         1, 1, 2,
         1, 1, 2]

        Pairwise differences (j-k):

        j\k  1  1  2  1  1  2  1  1  2
        1    0  0 -1  0  0 -1  0  0 -1
        1    0  0 -1  0  0 -1  0  0 -1
        2    1  1  0  1  1  0  1  1  0
        1    0  0 -1  0  0 -1  0  0 -1
        1    0  0 -1  0  0 -1  0  0 -1
        2    1  1  0  1  1  0  1  1  0
        1    0  0 -1  0  0 -1  0  0 -1
        1    0  0 -1  0  0 -1  0  0 -1
        2    1  1  0  1  1  0  1  1  0
        """

        # Replace all non-zero values with -10000
        # - This blocks attention across different regions
        """
        Replace all non-zero values   with -10000
        - This blocks attention across different regions
        - Final visual representation for the same example:

        0 0 X 0 0 X 0 0 X
        0 0 X 0 0 X 0 0 X
        X X 0 X X 0 X X 0
        0 0 X 0 0 X 0 0 X
        0 0 X 0 0 X 0 0 X
        X X 0 X X 0 X X 0
        0 0 X 0 0 X 0 0 X
        0 0 X 0 0 X 0 0 X
        X X 0 X X 0 X X 0
        """
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-10000.0)).masked_fill(attn_mask == 0, 0.0)

        """
        Final attention mask shape: (num_windows, win*win, win*win)
        Each window can attend only to positions with the same region ID
        """
        return attn_mask

    def forward(self, x):
    
        """
        x: (B, H*W, C)
        returns: (B, H*W, C)

        Standard Swin flow: the mask is used in the attention step only if shift > 0.
        """
    
        H, W = self.res
        B, L, C = x.shape
        assert L == H * W, f"Input length {L} doesn't match H*W={H*W}"

        residual = x

        # LN + reshape to image
        x = self.layer_norm1(x)   # (B, H*W, C)
        x = x.view(B, H, W, C)    # (B, H, W, C)

        # Cyclic shift (only if shift > 0)
        if self.shift > 0:
            # Shift up-left by shift
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # Partition windows
        x_windows = window_partition(x, self.win)  # (B*nW, win, win, C)
        x_windows = x_windows.contiguous().view(-1, self.win * self.win, C)  # (B*nW, win*win, C)

        # Attn + (optional) mask
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x_windows.device, dtype=x_windows.dtype)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (B*nW, win*win, C)

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.win, self.win, C)
        x = window_reverse(attn_windows, self.win, H, W)  # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        # Flatten back + residual
        x = x.contiguous().view(B, H * W, C)
        x = residual + x

        # MLP + residual
        x = x + self.mlp(self.layer_norm2(x))
        return x


class PatchMerging(nn.Module):
    
    """
    Patch Merging layer (Swin Transformer downsampling).

    Purpose:
      - Downsample spatial resolution by 2x in each dimension: (H, W) -> (H/2, W/2)
      - Increase channel dimension: C -> 2C  (after a linear projection)

    This is Swin’s “hierarchical” step (like pooling/strided conv in CNNs),
    but done by concatenating neighboring patch tokens.

    Input:
      x: (B, H*W, C)   tokens laid out on an HxW grid

    Output:
      x_out: (B, (H/2)*(W/2), 2C)
      H_out = H/2
      W_out = W/2

    Key idea (2x2 merge):
      Group each 2x2 block of tokens into one token:

        (r,c)     (r,c+1)
        (r+1,c)   (r+1,c+1)

      Concatenate their channels -> 4C, then LayerNorm(4C), then Linear(4C -> 2C).

    Concrete toy example (H=W=4):
      Original grid has 16 tokens:
        t00 t01 t02 t03
        t10 t11 t12 t13
        t20 t21 t22 t23
        t30 t31 t32 t33

      After patch merging, grid becomes 2x2 = 4 tokens:
        [t00,t10,t01,t11]   [t02,t12,t03,t13]
        [t20,t30,t21,t31]   [t22,t32,t23,t33]

      Each bracket means concatenation along channel dim => 4C.
    """

    def __init__(self, dim):
        """
        dim (int): input token embedding dimension C.
                   Example: if C=96, output dimension becomes 2C=192.
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.layer_norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert C == self.dim, f"Expected channel dim {self.dim}, got {C}"
        assert L == H * W, f"Input length {L} doesn't match H*W={H*W}"
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for PatchMerging"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, (H // 2) * (W // 2), 4 * C)

        x = self.layer_norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchEmbed(nn.Module):
    
    """
    Patch embedding using Conv2d.

    Converts image to tokens.

    Input:
      x: (B, in_chans, H_img, W_img)

    Output:
      tokens: (B, H_tok*W_tok, embed_dim)
      H_tok, W_tok: spatial token resolution after patchifying

    Example:
      H_img=W_img=224, patch=4 -> H_tok=W_tok=56
      H_img=W_img=28,  patch=2 -> H_tok=W_tok=14
    """

    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H_tok, W_tok)
        B, C, H_tok, W_tok = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H_tok*W_tok, embed_dim)
        return x, H_tok, W_tok


class BasicLayer(nn.Module):
    
    """
    One Swin stage (sometimes called BasicLayer in official code).

    A stage consists of:
      - depth SwinBlocks at a fixed resolution (H, W)
      - shift alternates: 0, win//2, 0, win//2, ...
      - optional PatchMerging at the end (downsample for the next stage)

    Inputs:
      x: (B, H*W, C)

    Outputs:
      x: (B, H'*W', C')
      H', W': updated resolution
    """

    def __init__(self, dim, res, depth, num_heads, win, mlp_ratio=4, downsample=True):
        super().__init__()
        self.dim = dim
        self.res = res
        self.depth = depth
        self.win = win
        self.downsample = downsample

        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else (win // 2)
            blocks.append(SwinBlock(dim=dim, res=res, win=win, shift=shift, num_heads=num_heads, mlp_ratio=mlp_ratio))
        self.blocks = nn.Sequential(*blocks)

        self.patch_merging = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        # Apply Swin blocks at fixed resolution
        x = self.blocks(x)

        # Downsample at end of stage if requested
        if self.patch_merging is not None:
            x, H, W = self.patch_merging(x, H, W)

        return x, H, W


class SwinTransformer(nn.Module):
    
    """
    Vanilla Swin Transformer architecture (4 stages).

    This version follows your code style:
      - tokens are (B, H*W, C)
      - attention uses NHWC reshape inside SwinBlock
      - PatchMerging downsamples between stages
      - Window size is constant across stages (default 7 like original Swin)

    Default parameters are Swin-T style:
      patch_size = 4
      embed_dim  = 96
      depths     = (2, 2, 6, 2)
      num_heads  = (3, 6, 12, 24)
      win        = 7
      mlp_ratio  = 4

    IMPORTANT:
      Your SwinBlock requires knowing res=(H,W) at construction time.
      For “vanilla Swin” on a fixed input size, we pass img_size at init
      so that resolutions are known in advance.

    Example (ImageNet-style):
      img_size=224, patch=4 -> stage1 res = 56x56
      stage2 res = 28x28
      stage3 res = 14x14
      stage4 res = 7x7

    Example (MNIST-like):
      img_size=28, patch=2 -> stage1 res = 14x14
      stage2 res = 7x7
      stage3 res = 3x3 (not divisible by win=7!) -> so for MNIST you usually DO NOT use 4 stages
      or you must change patch_size/win/depths to be consistent.
    """

    def __init__(
        self,
        img_size=224,
        in_channels=3,
        num_classes=1000,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        win=7,
        mlp_ratio=4
    ):
        super().__init__()

        assert len(depths) == 4 and len(num_heads) == 4, "Vanilla Swin uses 4 stages by default."
        self.img_size = img_size
        self.patch_size = patch_size
        self.win = win

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=in_channels, embed_dim=embed_dim, patch_size=patch_size)

        # Compute token resolution after patch embedding
        # H_tok = img_size / patch_size
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        H0 = img_size // patch_size
        W0 = img_size // patch_size

        # Channel dims per stage: C, 2C, 4C, 8C
        dims = [embed_dim, 2 * embed_dim, 4 * embed_dim, 8 * embed_dim]

        # Resolutions per stage: (H0, W0), (H0/2, W0/2), (H0/4, W0/4), (H0/8, W0/8)
        res1 = (H0, W0)
        res2 = (H0 // 2, W0 // 2)
        res3 = (H0 // 4, W0 // 4)
        res4 = (H0 // 8, W0 // 8)

        # Sanity checks for window partitioning and patch merging
        for r in [res1, res2, res3, res4]:
            assert r[0] > 0 and r[1] > 0, f"Invalid stage resolution {r}"
            assert r[0] % win == 0 and r[1] % win == 0, (
                f"Stage resolution {r} must be divisible by win={win} (or add padding)."
            )

        # Build 4 stages
        self.stage1 = BasicLayer(dims[0], res1, depths[0], num_heads[0], win, mlp_ratio, downsample=True)
        self.stage2 = BasicLayer(dims[1], res2, depths[1], num_heads[1], win, mlp_ratio, downsample=True)
        self.stage3 = BasicLayer(dims[2], res3, depths[2], num_heads[2], win, mlp_ratio, downsample=True)
        self.stage4 = BasicLayer(dims[3], res4, depths[3], num_heads[3], win, mlp_ratio, downsample=False)

        # Final norm + head
        self.final_norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x):

        """
        x: (B, in_chans, img_size, img_size)

        returns:
          logits: (B, num_classes)
        """

        # Patch embedding
        # (B, C_in, H, W) -> (B, H0*W0, C)
        x, H, W = self.patch_embed(x)

        # Stage 1: (H0, W0, C)
        x, H, W = self.stage1(x, H, W)  # after merging: H/2, W/2, C*2

        # Stage 2
        x, H, W = self.stage2(x, H, W)  # after merging: H/2, W/2, C*2

        # Stage 3
        x, H, W = self.stage3(x, H, W)  # after merging: H/2, W/2, C*2

        # Stage 4 (no merging)
        x, H, W = self.stage4(x, H, W)

        # Final norm in token space
        x = self.final_norm(x)  # (B, L, C_last)

        # Global average pooling over tokens
        x = x.mean(dim=1)       # (B, C_last)

        # Classifier
        x = self.head(x)        # (B, num_classes)
        return x



class SwinTransformerTiny(nn.Module):

    """
    SwinTransformerTiny: a tiny, MNIST-friendly Swin using non-canonical building blocks.

    Designed for small images (e.g. MNIST 28x28), where the full 4-stage Swin hierarchy
    is not feasible without resizing/padding.

    Typical MNIST configuration:
      img_size=28
      patch_size=2  -> tokens are 14x14
      win=7         -> windows fit 14x14 and 7x7 exactly
      stage1: res=14x14, dim=embed_dim
      PatchMerging -> res=7x7, dim=2*embed_dim
      stage2: res=7x7, dim=2*embed_dim
      head: LayerNorm + global average pooling + linear classifier

    Notes:
      - This avoids stages 3/4 entirely (no illegal 3x3 or 1x1 resolutions).
      - It preserves your rule: stage resolution must be divisible by win.
    """

    def __init__(
        self,
        img_size=28,
        in_channels=1,
        num_classes=10,
        patch_size=2,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(3, 6),
        win=7,
        mlp_ratio=4
    ):
        super().__init__()

        assert len(depths) == 2 and len(num_heads) == 2, "Tiny Swin uses exactly 2 stages"
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=in_channels, embed_dim=embed_dim, patch_size=patch_size)

        # Token resolution after patch embed
        H0 = img_size // patch_size
        W0 = img_size // patch_size

        # Stage resolutions
        res1 = (H0, W0)           # e.g. 14x14
        res2 = (H0 // 2, W0 // 2) # e.g. 7x7

        # Sanity checks (your SwinBlock assumes clean window partition)
        assert res1[0] % win == 0 and res1[1] % win == 0, f"Stage1 res {res1} must be divisible by win={win}"
        assert res2[0] % win == 0 and res2[1] % win == 0, f"Stage2 res {res2} must be divisible by win={win}"

        # Stage 1: dim = embed_dim, downsample at end -> dim becomes 2*embed_dim
        self.stage1 = BasicLayer(
            dim=embed_dim,
            res=res1,
            depth=depths[0],
            num_heads=num_heads[0],
            win=win,
            mlp_ratio=mlp_ratio,
            downsample=True
        )

        # Stage 2: dim = 2*embed_dim, no downsample
        self.stage2 = BasicLayer(
            dim=2 * embed_dim,
            res=res2,
            depth=depths[1],
            num_heads=num_heads[1],
            win=win,
            mlp_ratio=mlp_ratio,
            downsample=False
        )

        # Final norm + head
        self.final_norm = nn.LayerNorm(2 * embed_dim)
        self.head = nn.Linear(2 * embed_dim, num_classes)

    def forward(self, x):

        """
        Inputs:
            x: (B, in_chans, img_size, img_size)

        Returns:
            logits: (B, num_classes)
        """

        # Patch embedding:
        # (B, C_in, H, W) -> (B, H0*W0, embed_dim), plus token grid (H0, W0)
        x, H, W = self.patch_embed(x)

        # Stage 1 (14x14 -> PatchMerging -> 7x7, channels x2)
        x, H, W = self.stage1(x, H, W)

        # Stage 2 (7x7, no downsample)
        x, H, W = self.stage2(x, H, W)

        # Final norm in token space
        x = self.final_norm(x)      # (B, L, 2*embed_dim)

        # Global average pooling over tokens
        x = x.mean(dim=1)           # (B, 2*embed_dim)

        # Classifier head
        x = self.head(x)            # (B, num_classes)
        return x
