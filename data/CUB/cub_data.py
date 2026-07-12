from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import clip
import pickle
import os
from tqdm import tqdm
from utils import *
from args import get_args


class Processed_CUB_Dataset(Dataset):
    def __init__(
        self,
        args,
        data_root,
        split,
        meta_root=None,
        attr_name=None,
        src_dm_texts=None,
        tgt_dm_texts=None
    ):

        # ===== 1. load annotations =====
        if split == "train":
            self.annos = open(os.path.join(meta_root, "cub_train.txt")).readlines()
        elif split == "test":
            self.annos = open(os.path.join(meta_root, "cub_test.txt")).readlines()
        else:
            raise ValueError("split must be train or test")

        with open(os.path.join(meta_root, "class_avg_attribute.pkl"), "rb") as f:
            self.class_avg_attribute = pickle.load(f)
            self.cls_avg_concept = args.class_avg_concept

        self.data_root = data_root
        device = args.device

        # ===== 2. load CLIP =====
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)

        # ===== 3. class / concept =====
        with open(data_root.replace("images", "classes.txt"), "r") as f:
            self.classname2id = {
                x.split(" ")[1][4:].replace("_", " ").lower().rstrip(): (int(x.split(" ")[0]) - 1)
                for x in f.readlines()
            }

        with open(os.path.join(meta_root, "cub_concepts.txt"), "r") as f:
            self.concept2id = {
                x.rstrip(): c_id for c_id, x in enumerate(f.readlines())
            }

        # ===== 4. domain diffs + weights =====
        self.domain_diffs = None
        self.domain_weights = None

        if src_dm_texts is not None and tgt_dm_texts is not None:
            print("----------Computing Domain Differences + Weights----------")

            diffs_list = []
            weight_list = []

            for src_prompts, tgt_prompts in tqdm(
                zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)
            ):
                tqdm.write(f"{tgt_prompts} - {src_prompts}")

                source_embeddings, target_embeddings = get_domain_text_embs(
                    self.clip_model,
                    [src_prompts],
                    [tgt_prompts],
                    list(self.classname2id.keys()),
                    device
                )

                # normalize
                source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
                target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)

                # raw diff
                raw_diffs = target_embeddings.float() - source_embeddings.float()

                if raw_diffs.norm() == 0:
                    print("Warning: zero diff detected")

                # normalized diff
                normed_diffs = raw_diffs / (raw_diffs.norm(dim=-1, keepdim=True) + 1e-8)

                # ===== reliability score =====
                cos_sim = torch.matmul(normed_diffs, normed_diffs.T)

                mask = ~torch.eye(
                    cos_sim.size(0),
                    dtype=torch.bool,
                    device=cos_sim.device
                )

                reliability_score = cos_sim[mask].mean()

                diffs_list.append(normed_diffs)
                weight_list.append(reliability_score)

            # ===== stack =====
            self.domain_diffs = torch.stack(diffs_list, dim=0).to(device)

            scores = torch.stack(weight_list).to(device)

            # ===== 二值化权重 =====
            threshold = scores.median()

            self.domain_weights = torch.where(
                scores >= threshold,
                torch.tensor(1.8, device=device),
                torch.tensor(0.75, device=device)
            )

            # print("domain_diffs shape:", self.domain_diffs.shape)
            # print("domain_weights shape:", self.domain_weights.shape)
            # print("domain_weights sample:", self.domain_weights[:5])

        # 释放 CLIP（节省显存）
        self.clip_model = None

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label, top_x, top_y, btm_x, btm_y = self.annos[idx].strip().split(",")[0:6]

        image = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
        image = self.preprocess(image)

        label = int(cls_label) - 1

        # CUB 没有 concept label
        attr_label = torch.tensor([0] * len(self.concept2id))

        return image, label, attr_label


class Processed_CUBP_Dataset(Dataset):
    def __init__(self, args, data_root, split, meta_root=None, classname2id=None,concept2id=None):

        if split == "train":
            raise Exception("No processed data for train split")
        elif split == "test":
            self.annos = open(os.path.join(meta_root, "cubp_test.txt")).readlines()
        self.data_root = data_root
        device = args.device
        _, self.preprocess = clip.load(args.CLIP_type, device=device)
        self.classname2id = concept2id
        self.concept2id = concept2id


    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label = self.annos[idx].strip().split(",")[0:2]

        image = Image.open(os.path.join(self.data_root,img_path)).convert("RGB")
        image = self.preprocess(image)
        label = int(cls_label)-1
        ## target source has no concept label
        attr_label = torch.tensor([0]*77)

        return image, label, attr_label


class Processed_CUB_Dataset_Labo(Dataset):
    def __init__(self, args, data_root, split, meta_root=None,attr_name=None,src_dm_texts=None,tgt_dm_texts =None):
        if split == "train":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_train_all_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_train_all.pth"))
        elif split == "test":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_test_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_test.pth"))
        else:
            raise ValueError("split must be train or test")

        self.data_root = data_root
        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)
        with open("G:\DATA\DomainAdaptation\CUB\CUB_200_2011\classes.txt", "r") as f:
            self.classname2id = {x.split(" ")[1][4:].replace("_"," ").lower().rstrip():(int(x.split(" ")[0])-1) for x in f.readlines()}
        with open(os.path.join(meta_root,"cub_conceptNet_concepts.txt"),"r") as f:
            self.concept2id = {x.rstrip():c_id for c_id, x in enumerate(f.readlines())}
        # 改动时间26.4.29-----------------------------------
        if src_dm_texts is not None and tgt_dm_texts is not None:
            self.domain_diffs = []
            self.domain_weights = []

            print("----------Computing Domain Differences----------")

            for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
                tqdm.write(tgt_prompts + " - " + src_prompts)

                source_embeddings, target_embeddings = get_domain_text_embs(
                    self.clip_model,
                    [src_prompts],
                    [tgt_prompts],
                    list(self.classname2id.keys()),
                    device
                )

                source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
                target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)

                raw_diffs = target_embeddings.float() - source_embeddings.float()

                if raw_diffs.norm() == 0:
                    print(raw_diffs)

                normed_diffs = raw_diffs / (raw_diffs.norm(dim=-1, keepdim=True) + 1e-8)

                # reliability score:
                # 用不同类别之间 domain shift 方向的平均 cosine similarity 衡量可靠性。
                # 如果同一个 prompt 在不同类别上产生的偏移方向越一致，
                # reliability_score 越大，说明这个 prompt 越可靠。
                cos_sim = torch.matmul(normed_diffs, normed_diffs.T)

                # 去掉对角线，因为每个类别和自己的 cosine similarity 恒为 1
                mask = ~torch.eye(
                    cos_sim.size(0),
                    dtype=torch.bool,
                    device=cos_sim.device
                )

                reliability_score = cos_sim[mask].mean()

                self.domain_diffs.append(normed_diffs)
                self.domain_weights.append(reliability_score)

            self.domain_diffs = torch.stack(self.domain_diffs, dim=0).to(device)

            scores = torch.stack(self.domain_weights).to(device)
            threshold = scores.median()

            self.domain_weights = torch.where(
                scores >= threshold,
                torch.tensor(1.5, device=device),
                torch.tensor(0.5, device=device)
            )
        # 改动时间26.4.29-----------------------------------
        #
        # if src_dm_texts is not None and tgt_dm_texts is not None:
        #     self.domain_diffs = []
        #     print("----------Computing Domain Differences----------")
        #     for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
        #         tqdm.write(tgt_prompts+" - "+src_prompts)
        #         source_embeddings, target_embeddings = get_domain_text_embs(self.clip_model, [src_prompts], [tgt_prompts],
        #                                                                     list(self.classname2id.keys()),device)
        #         source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
        #         target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)
        #         diffs = target_embeddings.float() - source_embeddings.float()
        #         if diffs.norm() == 0:
        #             print(diffs)
        #         diffs /= diffs.norm(dim=-1, keepdim=True)
        #         self.domain_diffs.append(diffs)
        #     self.domain_diffs = torch.stack(self.domain_diffs, dim=0).to(device)
        # self.clip_model = None

    def __len__(self):
        return len(self.img_features)

    def __getitem__(self, idx):
        img_feature = self.img_features[idx]
        label = self.img_labels[idx]
        attr_label = torch.tensor([0] * len(self.concept2id))
        return img_feature, label, attr_label

class Processed_CUBP_Dataset_Labo(Dataset):
    def __init__(self, args, data_root, split, meta_root=None, classname2id=None,concept2id=None):
        ###
        if split == "train":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_train_all_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_train_all.pth"))
        elif split == "test":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_test_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_test.pth"))
        self.data_root = data_root
        device = args.device
        _, self.preprocess = clip.load(args.CLIP_type, device=device)
        self.classname2id = concept2id
        self.concept2id = concept2id


    def __len__(self):
        return len(self.img_features)

    def __getitem__(self, idx):
        img_feature = self.img_features[idx]
        label = self.img_labels[idx]
        attr_label = torch.tensor([0] * len(self.concept2id))
        return img_feature, label, attr_label


if __name__ == '__main__':
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    from domain_prompt import source_text_prompts, target_text_prompts

    train_dataset = Processed_CUB_Dataset(args,
                                          data_root="G:/DATA/DomainAdaptation/CUB/CUB_200_2011/images",
                                          split="train",
                                          meta_root="data/CUB",
                                          attr_name="CUBpath2attr.pkl",
                                          src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts)
    class2attribute = {}
    for idx in tqdm(range(len(train_dataset))):
        image, label, attr_label = train_dataset[idx]
        if label not in class2attribute:
            class2attribute[label] = [attr_label]
        else:
            class2attribute[label].append(attr_label)
    class_avg_attribute = {k: torch.stack(v, dim=0).float().mean(0) for k, v in class2attribute.items()}
    with open("data/CUB/class_avg_attribute.pkl", "wb") as f:
        pickle.dump(class_avg_attribute, f)
    print(0)

