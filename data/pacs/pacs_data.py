from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import clip
from reliability import attach_domain_reliability


class PACSDomainDataset(Dataset):
    """通用单域对数据集 (PACS / Office-Home): txt 格式 '相对路径,label'（label 0 基）"""

    def __init__(self, args, data_root, anno_file, classname2id, concept2id,
                 src_dm_texts=None, tgt_dm_texts=None, need_reliability=False):
        self.data_root = data_root
        anno_path = anno_file if os.path.exists(anno_file) else os.path.join('data/pacs', anno_file)
        self.annos = [l.strip() for l in
                      open(anno_path, encoding='utf-8').readlines() if l.strip()]
        self.classname2id = classname2id
        self.concept2id = concept2id
        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)
        if need_reliability:
            attach_domain_reliability(
                self, clip_model=self.clip_model,
                src_dm_texts=src_dm_texts, tgt_dm_texts=tgt_dm_texts,
                class_names=[n for n, _ in sorted(classname2id.items(), key=lambda kv: kv[1])],
                device=device, weight_strategy="prior_residual")
        self.clip_model = None

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        rel, label = self.annos[idx].split(',')
        image = Image.open(os.path.join(self.data_root, rel)).convert('RGB')
        image = self.preprocess(image)
        attr = torch.zeros(len(self.concept2id))
        return image, int(label), attr


class OfficeHomeDomainDataset(PACSDomainDataset):
    """Office-Home: 同 PACS 格式（4 域 × 65 类）"""
    pass
