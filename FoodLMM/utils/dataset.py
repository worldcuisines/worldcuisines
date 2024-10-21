import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

import utils.conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from .S2_f103_dataset import S2F103Dataset
from .S2_n5k_dataset import S2N5kDataset
from .VireoFood172_dataset_title import VireoFood172Dataset
from .ChineseFoodNet_dataset import ChineseFoodNet
from .food101_dataset import Food101Dataset 
from .data_processing import get_mask_from_json
from .nutrition5k_dataset import Nutrition5kDataset
from .recipe1M_dataset import Recipe1MDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, normalize_to_one, is_main_process, print_in_file, convert_gray_to_color_mask,
                    overlay_rgba_on_rgb)


def collate_fn(
    batch, tokenizer=None, conv_type="conv_food_assistant", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    mass_gt_list = []
    calorie_gt_list = []
    fat_gt_list = []
    carb_gt_list = []
    protein_gt_list = []
    total_mass_gt_list = []
    total_calorie_gt_list = []
    total_fat_gt_list = []
    total_carb_gt_list = []
    total_protein_gt_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        mass_gt,
        calorie_gt,
        fat_gt,
        carb_gt,
        protein_gt,
        total_mass_gt,
        total_calorie_gt,
        total_fat_gt,
        total_carb_gt,
        total_protein_gt,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        mass_gt_list.append(mass_gt)
        calorie_gt_list.append(calorie_gt)
        fat_gt_list.append(fat_gt)
        carb_gt_list.append(carb_gt)
        protein_gt_list.append(protein_gt)
        total_mass_gt_list.append(total_mass_gt)
        total_calorie_gt_list.append(total_calorie_gt)
        total_fat_gt_list.append(total_fat_gt)
        total_carb_gt_list.append(total_carb_gt)
        total_protein_gt_list.append(total_protein_gt)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    # print(masks)
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1" or conv_type == "conv_food_assistant":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    temp = 0
    #
    # q_file = './train_example/training_s2_yyh_select.txt'
    # logfile_q = open(q_file, 'a')
    # print_in_file(
    #     content='------------------- {} -------------------{} * {}: {} \n'.format(image_path_list, '\n',
    #                                                                               '[conv]',
    #                                                                               conversation_list), file=logfile_q)
    # print_in_file('segment_num={}'.format(len(masks_list)), file=logfile_q)
    # print_in_file(' ', file=logfile_q)

    idx_im = 0
    for conversation, target in zip(conversation_list, targets):
        # print(conversation)

        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        # q_file = './train_example/training.txt'
        # logfile_q = open(q_file, 'w')
        # print_in_file(
        #     content='------------------- {} -------------------{} * {}: {} \n'.format(image_path_list[idx_im], '\n',
        #                                                                               '[conversation]',
        #                                                                               conversation), file=logfile_q)
        # print_in_file(' ', file=logfile_q)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):

            # q_file = './train_example/training.txt'
            # logfile_q = open(q_file, 'w')
            # print_in_file(
            #     content='------------------- {} -------------------{} * {}: {} \n'.format(image_path_list[idx_im], '\n', '[rou]', rou), file=logfile_q)
            # print_in_file(' ', file=logfile_q)
            #
            # save_img = None
            # image_np = cv2.imread(image_path_list[idx_im])
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # colored_mask = torch.zeros(size=image_np.shape)
            #
            # if masks_list[idx_im].shape[0] > 0:
            #     print('num_masks:', masks_list[idx_im].shape[0])
            #     for idx in range(masks_list[idx_im].shape[0]):
            #         print('masks_list[{}]'.format(idx_im), masks_list[idx_im].shape)
            #         pred_mask = masks_list[idx_im][idx, :, :]
            #         pred_mask = pred_mask.detach().cpu().numpy()
            #         pred_mask = pred_mask > 0
            #         print('pred_mask.shape', pred_mask.shape)
            #         colored_mask += convert_gray_to_color_mask(pred_mask, color_idx=idx)
            #     save_path = "{}/{}_{}_mask.jpg".format(
            #         './train_example', image_path_list[idx_im].split('/')[-1].split('.jpg')[0], idx_im
            #     )
            #     colored_mask_numpy = colored_mask.cpu().numpy()
            #     colored_mask_numpy = (colored_mask_numpy * 255).astype(np.uint8)
            #     save_img = cv2.cvtColor(colored_mask_numpy, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(save_path, save_img)
            #     # [yyh] save the colored multi-mask img
            #     save_img = overlay_rgba_on_rgb(torch.tensor(image_np), torch.tensor(colored_mask) * 255)
            #     save_img = cv2.cvtColor(save_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            #     save_path = "{}/{}_{}_colored.jpg".format(
            #             './train_example', image_path_list[idx_im].split('/')[-1].split('.jpg')[0], idx_im
            #     )
            #     cv2.imwrite(save_path, save_img)
            #     print("{} has been saved.".format(save_path))
            #     save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)

            if rou == "":
                break
            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            # print('parts', parts)

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        temp += 1

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
        idx_im += 1

    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            q_file = './train_samples_examples/over_length_dialog.txt'
            logfile_q = open(q_file, 'a')
            print_in_file(
                content=' \n \n------------------- {}->{} -------------------{} * {}: {} \n -> \n {}  \n \n'.format(input_ids.shape[1], truncate_len, '\n', '[dialog]',
                                                                                       conversation_list, ''.join(conversation_list)[:truncate_len]), file=logfile_q)

            print('input_ids.shape[1]', input_ids.shape[1], '->', truncate_len)

            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "mass_gt_list": mass_gt_list,
        "calorie_gt_list": calorie_gt_list,
        "fat_gt_list": fat_gt_list,
        "carb_gt_list": carb_gt_list,
        "protein_gt_list": protein_gt_list,
        "total_mass_gt_list": total_mass_gt_list,
        "total_calorie_gt_list": total_calorie_gt_list,
        "total_fat_gt_list": total_fat_gt_list,
        "total_carb_gt_list": total_carb_gt_list,
        "total_protein_gt_list": total_protein_gt_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
            self,
            base_image_dir,
            tokenizer,
            vision_tower,
            samples_per_epoch: int = 500 * 8 * 2 * 10,
            image_size: int = 224,
            seg_num: int = 20,
            num_OOD_per_sample: int = 0,
            dataset="sem_seg||vqa",
            sample_rate=None,
            sem_seg_data="foodseg103||uecfoodpix",
            vqa_data="recipe1m||nutrition5k||VieroFood172",
            ft=False,
            model_name=''
    ):
        if ft:
            print('fine-tune mode')
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.samples_per_epoch = samples_per_epoch
        self.datasets = dataset.split("||")
        self.all_datasets = []
        self.all_datasets_name = ''
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets_name += "sem_seg"
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir=base_image_dir,
                        vision_tower=vision_tower,
                        image_size=image_size,
                        seg_num=seg_num,
                        num_OOD_per_sample=num_OOD_per_sample,
                        sem_seg_data=sem_seg_data,
                        sem_seg_data_ratio=[0.7, 0.3],
                        is_val=False,
                        ft=ft
                    )
                )
            elif dataset == "vqa":
                if len(self.all_datasets_name) > 2:
                    self.all_datasets_name += "||"
                self.all_datasets_name += vqa_data
                for sub_dataset in vqa_data.split("||"):
                    print(sub_dataset)
                    if sub_dataset == 'VieroFood172':
                        self.all_datasets.append(
                            VireoFood172Dataset(
                                base_image_dir=base_image_dir,
                                vision_tower=vision_tower,
                                image_size=image_size,
                            )
                        )
                    elif sub_dataset == 'nutrition5k':
                        self.all_datasets.append(
                            Nutrition5kDataset(
                                base_image_dir=base_image_dir,
                                vision_tower=vision_tower,
                                image_size=image_size,
                                seg_num=seg_num,
                                ft=ft
                            )
                        )
                    elif sub_dataset == 'recipe1m':
                        self.all_datasets.append(
                            Recipe1MDataset(
                                base_image_dir=base_image_dir,
                                vision_tower=vision_tower,
                                image_size=image_size,
                            )
                        )
                    elif sub_dataset == 'food101':
                        self.all_datasets.append(
                            Food101Dataset(
                                base_image_dir=base_image_dir,
                                vision_tower=vision_tower,
                                image_size=image_size,
                            )
                        )
                    elif sub_dataset == 'ChineseFoodNet':
                        self.all_datasets.append(
                            ChineseFoodNet(
                                base_image_dir=base_image_dir,
                                vision_tower=vision_tower,
                                image_size=image_size,
                            )
                        )
            elif dataset == "mr_conversation":
                if len(self.all_datasets_name) > 2:
                    self.all_datasets_name += "||"
                self.all_datasets_name += 'S2_n5k'
                self.all_datasets.append(
                    S2N5kDataset(
                        base_image_dir=base_image_dir,
                        vision_tower=vision_tower,
                        image_size=image_size
                    )
                )
            elif dataset == "reason_seg":
                if len(self.all_datasets_name) > 2:
                    self.all_datasets_name += "||"
                self.all_datasets_name += 'S2_f103'
                self.all_datasets.append(
                    S2F103Dataset(
                        base_image_dir=base_image_dir,
                        vision_tower=vision_tower,
                        image_size=image_size,
                        seg_num=seg_num,
                        model_name=model_name
                    )
                )

        sample_nums = []
        for dataset in self.all_datasets:
            sample_nums.append(len(dataset))
        if sample_rate is None or len(sample_rate) == 0:
            sample_rate = sample_nums
        self.sample_rate = normalize_to_one(sample_rate[:len(self.all_datasets)])
        flag = True
        while flag:
            for i in range(len(self.sample_rate)):
                if self.sample_rate[i] < 0.1:
                    self.sample_rate[i] += 0.1
            self.sample_rate = normalize_to_one(self.sample_rate)
            flag = False
            for i in range(len(self.sample_rate)):
                if self.sample_rate[i] < 0.1:
                    flag = True

        self.all_sample_nums = sum(sample_nums)

        if is_main_process():
            rate_str = ''
            sample_str = ''
            dataset_names = self.all_datasets_name.split('||')
            for ds in range(len(dataset_names)):
                rate_str += '{}={}, '.format(dataset_names[ds], self.sample_rate[ds])
                sample_str += '{}={}, '.format(dataset_names[ds], sample_nums[ds])
            print('[ALL_dataset]', self.all_datasets_name)
            print('[train_samples]', sample_str)
            print('[sample_rate]', rate_str)
            print('[ALL_training data]',  self.all_sample_nums)

    def __len__(self):
        return min(self.samples_per_epoch, self.all_sample_nums)

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.all_datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        return data[0]
