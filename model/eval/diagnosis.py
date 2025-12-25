import argparse
import itertools
import json
import os
import pdb
import random
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm

ds_collections = {
    'diagnosis': {
        'root': '/home/liyh/dataset/mimic-cxr-png/',
        'annotation':'/home/liyh/InternVL-main/internvl_chat/data/diagnosisi/mimic_test_diagnosis.json',
        'max_new_tokens': 90,
        'min_new_tokens': 1,
    }
}



import json
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score

def calculate_metrics(results_file, gts_file):
    # 读取预测结果和真实标签
    with open(results_file, 'r') as f:
        results = json.load(f)
    with open(gts_file, 'r') as f:
        gts = json.load(f)

    # 将结果和标签按image_id组织成字典
    results_dict = {int(result['image_id']): result['caption'] for result in results}
    gts_dict = {int(gt['image_id']): gt['label'] for gt in gts}

    # 提取预测结果和真实标签
    true_labels = []
    predicted_labels = []
    for image_id in results_dict:
        if image_id in gts_dict:
            true_labels.append(gts_dict[image_id])
            predicted_labels.append(results_dict[image_id])

    # 计算准确率、召回率和F1分数
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, name, root, annotation, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.images = json.load(open(annotation))
        self.name = name
        self.prompt = prompt
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        # return 2000
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]['id']
        image_path = self.images[idx]['image']
        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path)
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        lable = self.images[idx]['caption']

        return {
            'image_id': image_id,
            'input_text': self.prompt,
            'pixel_values': pixel_values,
            'lables':lable,
        }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    lables = [_['lables'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, image_ids,input_tokens.input_ids, input_tokens.attention_mask,lables


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'no finding'
]
def map_conditions(conditions_str):
    conditions_list = [condition.strip().lower() for condition in conditions_str.split(',')]
    return [1 if condition in conditions_list else 0 for condition in CONDITIONS]

def evaluate_chat_model():
    prompt0 = "<image>\nPlease review the following chest X-ray image and assess for any signs of an enlarged cardiomediastinum. An enlarged cardiomediastinum refers to the abnormal widening of the mediastinum, typically due to issues such as aortic disease, lymphadenopathy, or masses within the mediastinal region. Imaging characteristics of an enlarged cardiomediastinum include a mediastinum width greater than 8 cm on a PA X-ray or 6 cm on an AP X-ray, and loss of clear mediastinal borders. If an enlarged cardiomediastinum is present, output 1, otherwise, output 0."
    prompt1 = "<image>\nPlease review the following chest X-ray image and assess for any signs of cardiomegaly. Cardiomegaly refers to an abnormal enlargement of the heart, often caused by myocardial hypertrophy or chamber dilation, and may result from conditions like hypertension, cardiomyopathy, or valvular disease. Imaging characteristics of cardiomegaly include a cardiothoracic ratio greater than 0.5, widening of the cardiac silhouette, and prominence of the left or right ventricle. If cardiomegaly is present, output 1,otherwise, output 0."
    prompt2 = "<image>\nPlease review the following chest X-ray image and assess for any signs of lung opacity. Lung opacity refers to areas on the X-ray where normal lung transparency is reduced, indicating potential abnormalities like fluid accumulation, infection, or inflammation. Typical signs of lung opacity include regions of increased density that obscure underlying structures, such as blood vessels, or a 'ground-glass' appearance. If lung opacity is present, output 1; otherwise, output 0."
    prompt3 = "<image>\nPlease review the following chest X-ray image and assess for any signs of a lung lesion. A lung lesion refers to an abnormal area within the lung that may appear as a localized spot or mass, indicating possible tumors, nodules, or other localized abnormalities. Key imaging characteristics of a lung lesion include a distinct, well-defined shadow or mass within the lung fields, which may vary in size and shape. If a lung lesion is present, output 1; otherwise, output 0."
    prompt4 = "<image>\nPlease review the following chest X-ray image and assess for any signs of pulmonary edema. Pulmonary edema refers to the accumulation of fluid within the lung tissue, often due to heart failure or fluid overload. Typical imaging signs of edema include diffuse haziness or 'bat wing' patterns around the central lung fields, blurred vascular outlines, and increased lung markings. If pulmonary edema is present, output 1; otherwise, output 0."
    prompt5 = "<image>\nPlease review the following chest X-ray image and assess for any signs of consolidation. Consolidation refers to a region of the lung where air has been replaced by fluid, pus, blood, or cells, often due to infection or inflammation. Imaging signs of consolidation include areas of increased density, loss of normal lung markings, and air bronchograms (visible air-filled bronchi within the opaque area). If consolidation is present, output 1; otherwise, output 0."
    prompt6 = "<image>\nPlease review the following chest X-ray image and assess for any signs of pneumonia. Pneumonia is an infection of the lung tissue that causes inflammation and may lead to fluid accumulation in the alveolar spaces. Common imaging characteristics of pneumonia include areas of consolidation or opacity, which may present as patchy or lobar densities, often with air bronchograms visible within the affected regions. If pneumonia is present, output 1; otherwise, output 0."
    prompt7 = "<image>\nPlease review the following chest X-ray image and assess for any signs of atelectasis. Atelectasis refers to the partial or complete collapse of a lung or a lobe, often resulting from obstruction or pressure. Key imaging signs of atelectasis include areas of increased density with volume loss, shift of surrounding structures (such as the trachea or mediastinum) towards the affected area, and elevation of the diaphragm on the affected side. If atelectasis is present, output 1; otherwise, output 0."
    prompt8 = "<image>\nPlease review the following chest X-ray image and assess for any signs of pneumothorax. Pneumothorax refers to the presence of air in the pleural space, which can lead to lung collapse. Key imaging characteristics of pneumothorax include the visibility of the visceral pleural line, a lack of vascular markings beyond the pleural line, and potential displacement of the mediastinum away from the affected side. If pneumothorax is present, output 1; otherwise, output 0."
    prompt9 = "<image>\nPlease review the following chest X-ray image and assess for any signs of pleural effusion. Pleural effusion refers to the accumulation of fluid in the pleural space, which can occur due to various conditions, including heart failure, infection, or malignancy. Key imaging characteristics of pleural effusion include blunting of the costophrenic angles, increased opacity in the affected hemithorax, and possible meniscus sign. If pleural effusion is present, output 1; otherwise, output 0."
    prompt10 = "<image>\nPlease review the following chest X-ray image and assess for any signs of other pleural conditions. This may include various abnormalities such as pleural thickening, pleural masses, or other non-fluid related issues in the pleural space. Key imaging characteristics to look for include irregular pleural margins, localized pleural thickening, or the presence of a pleural mass. If any other pleural abnormalities are present, output 1; otherwise, output 0."
    prompt11 = "<image>\nPlease review the following chest X-ray image and assess for any signs of fractures. A fracture refers to a break in the continuity of bone, which may involve the ribs or other structures within the thoracic region. Key imaging characteristics of fractures include visible breaks in the bone cortex, displacement of bone fragments, and associated soft tissue swelling or hematoma. If any fractures are present, output 1; otherwise, output 0."
    prompt12 = "<image>\nPlease review the following chest X-ray image and assess for any abnormalities. If there are no significant findings, output 1; otherwise, output 0."

    prompt = prompt1
    fadfa = 'cardiomegaly'
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        annotation = ds_collections[ds_name]['annotation']
        if type(annotation) == list:
            annotation = annotation[0]
        dataset = CaptionDataset(
            name=ds_name,
            root=ds_collections[ds_name]['root'],
            annotation=annotation,
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),

        )
        file_counter = 0
        image_ids, captions,gtss = [], [], []
        for _, (pixel_values, ids, _, _,lables) in tqdm(enumerate(dataloader)):
            file_counter += 1
            print(f"\n Processing file {file_counter}/{len(dataloader)},current id {ids}")
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
                verbose=True
            )
            image_ids.extend(ids)
            captions.extend([pred])
            gtss.extend(lables)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        merged_gts = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)
        torch.distributed.all_gather_object(merged_gts, gtss)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
        merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
        average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
        print(f'Average caption length: {average_length}')

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            results = []
            gts_img= []

            i = 0
            for image_id, caption,gt in zip(merged_ids, merged_captions,merged_gts):
                i += 1
                results.append({
                    'image_id':str(image_id),
                    'caption': caption
                })
                # condition_flags = map_conditions(gt)
                # condition_flags_str = json.dumps(condition_flags)
                if fadfa in gt:
                    aa = '1'
                else:
                    aa= '0'
                gts_img.append({
                    'image_id':str(image_id),
                    'label':aa
                })

            gts = gts_img

            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            # da = 'enlarged cardiomediastinum'
            results_file = f'{ds_name}_{fadfa}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(results, open(results_file, 'w'))

            gts_file = f'{"gts"}_{fadfa}.json'
            gts_file = os.path.join(args.out_dir, gts_file)
            json.dump(gts, open(gts_file, 'w'))

            summary = calculate_metrics(results_file,gts_file)
            print(summary)
            summaries.append([args.checkpoint, ds_name, average_length, summary,prompt])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='diagnosis')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
