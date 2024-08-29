# flame_feature_extractor
A simple library based on [lightning_track](https://github.com/xg-chu/lightning_track). 

Most of the code is taken from the original library and modified to suit my needs.

## Installation

```bash
git clone https://github.com/naba89/flame_feature_extractor.git
cd flame_feature_extractor
pip install -r requirements_pre.txt
pip install -r requirements_post.txt
bash build_resources.sh
```

## Usage

```python
import torch
import cv2
import numpy as np
from flame_feature_extractor.feature_extractor import PreProcessBatchFace, FeatureExtractorFLAME
from flame_feature_extractor.renderer import FlameRenderer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imgs = ...   # batch of RGB images of shape (B, 3, H, W) , where B is the batch size or could be frames of a video

preprocessor = PreProcessBatchFace()  # can be used in dataloader and run on CPU in parallel

feature_extractor = FeatureExtractorFLAME().to(device)  # can be used as part of a model and run on GPU

renderer = FlameRenderer(
    max_batch_size=128,
    fixed_transform=False,
    n_shape=100,
    n_exp=50,
    scale=5.0,
    device=device.type
)
preprocessor_output = preprocessor(imgs)

preprocessor_output = {k: v.to(device) for k, v in preprocessor_output.items()}
output = feature_extractor(mica_images=preprocessor_output['mica_images'], 
                           emoca_images=preprocessor_output['emoca_images'])

print(output['shape'].shape, output['expression'].shape, output['pose'].shape)

out_ims, _ = renderer.render_batch(**output)

# Save the first rendered image as an example, rendering is done in batch
cv2.imwrite('rendered.png', out_ims.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0])
```

## Reference
```bibtex
@inproceedings{
    chu2024gpavatar,
    title={{GPA}vatar: Generalizable and Precise Head Avatar from Image(s)},
    author={Xuangeng Chu and Yu Li and Ailing Zeng and Tianyu Yang and Lijian Lin and Yunfei Liu and Tatsuya Harada},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=hgehGq2bDv}
}
```
