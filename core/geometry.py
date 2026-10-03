import torch,os,sys
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, bilinear_sampler1d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

class Combined_Geo_Encoding_Volume:
    """幾何編碼體：把「聚合後代價體」與「左右特徵全對相關」各做成多層金字塔。
    推論時依每個像素目前的視差，在視差軸上取半徑 r 範圍內的值，作為 GRU 的幾何線索。
    """
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation：同一列上左圖每個像素與右圖每個像素的相似度
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.view(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        # 沿視差軸平均池化，建立多尺度金字塔(每層寬度減半)
        for _ in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)



    def __call__(self, disp, coords, dx, low_memory=True):
        # disp: 目前視差 (B,1,H,W)；coords: 像素 x 座標；dx: 取樣偏移 [-r..r]
        # low_memory=True 時用 1D 雙線性取樣，較省記憶體
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            with torch.profiler.record_function(f"make disp_lvl {i}"):
              geo_volume = self.geo_volume_pyramid[i]
              x0 = dx + disp.view(b*h*w, 1, 1, 1) / 2**i
            with torch.profiler.record_function(f"bilinear_sampler geo_volume {i}"):
              if low_memory:
                geo_volume = bilinear_sampler1d(geo_volume, x0, mode='bilinear', align_corners=True)
              else:
                y0 = torch.zeros_like(x0)
                disp_lvl = torch.cat([x0,y0], dim=-1)
                geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
              geo_volume = geo_volume.view(b, h, w, -1)   #(b, h, h, 3x3xC)

            with torch.profiler.record_function(f"make init_coords_lvl {i}"):
              init_corr = self.init_corr_pyramid[i]   # (B*H*W, 1, 1, W2)
              init_x0 = coords.view(b*h*w, 1, 1, 1)/2**i - disp.view(b*h*w, 1, 1, 1) / 2**i + dx   # X on right image
            with torch.profiler.record_function(f"bilinear_sampler init_corr {i}"):
              if low_memory:
                init_corr = bilinear_sampler1d(init_corr, init_x0, mode='bilinear', align_corners=True)
              else:
                init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
                init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
              init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)

        with torch.profiler.record_function(f"make out_pyramid"):
          out_pyramid = torch.cat(out_pyramid, dim=-1)
          return out_pyramid.permute(0, 3, 1, 2)   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2, normalize=True):
        # 計算同列相關：先對通道做 L2 正規化(即餘弦相似度)，再以 einsum 得到 (B,H,W1,1,W2)
        with torch.profiler.record_function("build corr"):
          B, D, H, W1 = fmap1.shape
          _, _, _, W2 = fmap2.shape
          fmap1 = fmap1.view(B, D, H, W1)
          fmap2 = fmap2.view(B, D, H, W2)
          if normalize:
            with torch.cuda.amp.autocast(enabled=False):
              corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
          else:
            corr = corr.view(B, H, W1, 1, W2).to(fmap1.dtype)
          corr = corr.view(B, H, W1, 1, W2).to(fmap1.dtype)
        return corr