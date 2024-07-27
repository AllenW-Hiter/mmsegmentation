from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DLOsSSDataset(BaseSegDataset):
    """DLOsSS dataset.

    DLOs semantic segmentation dataset. DLOs are 0 ,others is 255. 
    """
    METAINFO = dict(
        classes=("DLOs",),
        palette=[[255, 0, 0],])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_label.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)