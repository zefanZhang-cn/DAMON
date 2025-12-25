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
    'vqarad': {
        'root': "/home/liyh/dataset/VQA-RAD/",
        'annotation':"/home/liyh/InternVL-main/internvl_chat/data/single_image/vqarad_test.json",
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

    print(f"Accuracy: {accuracy}")

    return {'All_accuracy': accuracy}
def calculate_metrics11(results, gts,type = 'close'):
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
    print(f"{type}_Accuracy: {accuracy}")
    return {f'{type}_accuracy': accuracy}

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
        prompt = self.images[idx]['prompt']
        type = self.images[idx]['type']
        return {
            'image_id': image_id,
            'input_text': self.prompt,
            'pixel_values': pixel_values,
            'lables':lable,
            'question':prompt,
            'type':type
        }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    lables = [_['lables'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')
    questions = [_['question'] for _ in inputs]
    types = [_['type'] for _ in inputs]
    return pixel_values, image_ids,input_tokens.input_ids, input_tokens.attention_mask,lables,questions,types


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


def evaluate_chat_model():
    random.seed(args.seed)
    summaries = []
    prompt = ' dasda'
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
        typeaa = []
        for _, (pixel_values, ids, _, _,lables,questions,types) in tqdm(enumerate(dataloader)):
            prompt = questions[0]
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
            typeaa.extend(types)
            gtss.extend(lables)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        merged_gts = [None for _ in range(world_size)]
        merged_types = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)
        torch.distributed.all_gather_object(merged_gts, gtss)
        torch.distributed.all_gather_object(merged_types, typeaa)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
        merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
        merged_types = [_ for _ in itertools.chain.from_iterable(merged_types)]
        average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
        print(f'Average caption length: {average_length}')

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            results_close = []
            gts_img_close= []
            results_open = []
            gts_img_open = []
            i = 0
            for image_id, caption,gt,tt in zip(merged_ids, merged_captions,merged_gts,merged_types):
                if tt =='CLOSED':
                    i += 1
                    results_close.append({
                        'image_id':str(i*2),
                        'caption': caption.lower()
                    })
                    gts_img_close.append({
                        'image_id':str(i*2),
                        'label':gt.lower()
                    })
                else:
                    i += 1
                    results_open.append({
                        'image_id': str(i * 2),
                        'caption': caption.lower()
                    })
                    gts_img_open.append({
                        'image_id': str(i * 2),
                        'label': gt.lower()
                    })
            close_acc = calculate_metrics11(results_close,gts_img_close,'close')
            open_acc = calculate_metrics11(results_open,gts_img_open,'open')
            print(close_acc)
            print(open_acc)


            gts = gts_img_open+gts_img_close
            results = results_open+results_close
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}_{open_acc}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(results, open(results_file, 'w'))

            gts_file = f'{ds_name}_{"gts"}_{time_prefix}.json'
            gts_file = os.path.join(args.out_dir, gts_file)
            json.dump(gts, open(gts_file, 'w'))

            summary = calculate_metrics(results_file,gts_file)
            print(summary)
            summaries.append([args.checkpoint, ds_name, average_length, summary,close_acc,open_acc])

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
    parser.add_argument('--datasets', type=str, default='vqarad')
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
