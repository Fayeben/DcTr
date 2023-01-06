from .build import build_model_from_cfg
import sys
sys.path.append('../pointnet2_ops_lib')
sys.path.append('..')
import models.TopNet
import models.DcTr
import models.GRNet
import models.PCN
import models.FoldingNet
import models.Snowflake
import models.PMPnet
import models.ASFM
import models.VRCNet
import models.ecg
import models.cascade