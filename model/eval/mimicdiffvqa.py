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
    'mimicdiff': {
        'root': '/home/liyh/dataset/mimic-cxr-png/',
        'annotation': ['/home/liyh/InternVL-main/internvl_chat/data/mimicdiff/mimicdiff_test_label.json',
                       "/home/liyh/InternVL-main/internvl_chat/data/mimicdiff/mimicdiff_test_coco.json"
                       ],

        'max_new_tokens': 90,
        'min_new_tokens': 3,
    },
    'single': {
            'root': '/home/liyh/dataset/mimic-cxr-png/',
            'annotation': ['/home/liyh/InternVL-main/internvl_chat/data/mimicdiff/single_test.json',
                           "/home/liyh/InternVL-main/internvl_chat/data/mimicdiff/mimicdiff_test_coco.json"
                           ],

            'max_new_tokens': 90,
            'min_new_tokens': 1,
        }
}


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
        data_item = self.images[idx]
        imagestore, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            image_path = os.path.join(self.root, image_path)
            image = Image.open(image_path).convert('RGB')

            if self.dynamic_image_size:
                images = dynamic_preprocess(image, image_size=self.input_size,
                                            use_thumbnail=self.use_thumbnail,
                                            max_num=self.max_num // num_image)
                imagestore += images
                num_tiles.append(len(images))
            else:
                imagestore.append(image)
                num_tiles.append(1)
        lable = self.images[idx]['caption']
        prompt = self.images[idx]['prompt']
        # prompt = 'What has changed between the mian<image>\n and reference<image>\n x-ray image?'
        pixel_values = [self.transform(image) for image in imagestore]
        pixel_values = torch.stack(pixel_values)

        return {
            'image_id': image_id,
            'input_text': self.prompt,
            'pixel_values': pixel_values,
            'lables':lable,
            'prompt':prompt
        }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    lables = [_['lables'] for _ in inputs]
    prompts = [_['prompt'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, image_ids,input_tokens.input_ids, input_tokens.attention_mask,lables,prompts


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

    prompt = "What has changed between the main image<image>\n and the reference image<image>\n?"
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
        for _, (pixel_values, ids, _, _,lables,prompts) in tqdm(enumerate(dataloader)):
            file_counter += 1
            prompt = prompts[0]


            # print(prompt)
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

        assert len(merged_captions) == len(merged_gts), "两个列表的长度必须相同"
        # 创建一个字典列表，每个字典包含"label"和"caption"键
        combined_data = [{"label": merged_gts[i], "caption": merged_captions[i]} for i in range(len(merged_captions))]
        output_file_path = "base+all.json"
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到 {output_file_path}")



        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            results = []
            gts_img= []
            gts_ann =[]
            i = 100000
            for image_id, caption,gt in zip(merged_ids, merged_captions,merged_gts):
                i += 1
                results.append({
                    'image_id':str(i),
                    'caption': caption
                })
                gts_img.append({"id": str(i), 'caption': gt})
                gts_ann.append({"image_id": str(i), "id": str(i), 'caption': gt})

            gts = {'images': gts_img, 'annotations': gts_ann}

            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(results, open(results_file, 'w'))

            coco_file = f'{ds_name}_{"coco"}_{time_prefix}.json'
            coco_file = os.path.join(args.out_dir, coco_file)
            json.dump(gts, open(coco_file, 'w'))

            annotation = coco_file

            coco = COCO(annotation)
            coco_result = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()

            summary = coco_eval.eval.items()
            print(summary)
            summaries.append([args.checkpoint, ds_name, average_length, summary])

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
    parser.add_argument('--datasets', type=str, default='mimicdiff')
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
