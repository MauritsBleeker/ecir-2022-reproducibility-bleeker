import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
# from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import pickle
from six import BytesIO as IO
import random
import itertools
from functools import partial


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:
    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...
    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        roots['train'] = {'data': os.path.join(path, 'train_set.p')}
        roots['val'] = {'data': os.path.join(path, 'val_set.p')}
        roots['test'] = {'data': os.path.join(path, 'test_set.p')}

        ids = {'train': None, 'val': None, 'test': None}

    elif 'f30k' == name:
        #data_dir = os.path.join(path, 'images')

        roots['train'] = {'data': os.path.join(path, 'train_set.p')}
        roots['val'] = {'data': os.path.join(path, 'val_set.p')}
        roots['test'] = {'data':  os.path.join(path, 'test_set.p')}

        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids

class MultiModalDataset(data.Dataset):

    def __init__(self, data_file, split, vocab, transform=None, ranking_based=False, captions_per_img=5, n_sp=5):

        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = pickle.load(open(data_file['data'], 'rb'))
        self.caption_ids = list(self.dataset['captions'].keys())
        self.image_ids = list(self.dataset['images'].keys())

        self.captions_per_img = captions_per_img
        self.n_sp = n_sp
        self.ranking_based = ranking_based

    def __getitem__(self, index):
        """
        This function returns a tuple that is further passed to collate_fn
        """

        if self.ranking_based:
            img_id = self.image_ids[index]
            caption_ids = self.dataset['images'][img_id]['caption_ids']

            if len(caption_ids) > self.n_sp:
                caption_ids = random.sample(caption_ids, self.n_sp)

        else:
            caption_id = self.caption_ids[index]
            img_id = self.dataset['captions'][caption_id]['imgid']

        image = Image.open(IO(self.dataset['images'][img_id]['image'])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.

        if self.ranking_based:
            targets = []
            for caption_id in caption_ids[:self.captions_per_img]:
                caption, target = self.process_caption(caption_id)
                targets.append(target)
        else:
            caption, targets = self.process_caption(caption_id)

        return image, targets, index, img_id

    def __len__(self):
        if self.ranking_based:
            return len(self.image_ids)
        else:
            return len(self.caption_ids)

    def process_caption(self, caption_id):
        """
        :param caption_id:
        :return:
        """

        caption = self.dataset['captions'][caption_id]['caption']

        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower()
        )

        caption = []

        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        return caption, target


def collate_fn(data, ranking_based=False):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length

    if not ranking_based:
        data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, ids, img_ids = zip(*data)

    if ranking_based:
        captions = list(itertools.chain(*captions))

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_loader_single(data_name, split, data_file, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn, ranking_based=False, n_sp=5):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = MultiModalDataset(split=split,
                                data_file=data_file,
                                vocab=vocab,
                                transform=transform, ranking_based=ranking_based and split == 'train',
                                n_sp=n_sp)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=partial(collate_fn, ranking_based=ranking_based  and split == 'train'))
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):

        # Build Dataset Loader
    files, ids = get_paths(opt.data_path, data_name, opt.use_restval)

    transform = get_transform(data_name, 'train', opt)
    train_loader = get_loader_single(opt.data_name, 'train',
                                     files['train'],
                                     vocab, transform, ids=ids['train'],
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=workers,
                                     collate_fn=collate_fn, ranking_based=opt.ranking_based,  n_sp=opt.n_sp)

    transform = get_transform(data_name, 'val', opt)
    val_loader = get_loader_single(opt.data_name, 'val',
                                   files['val'],
                                   vocab, transform, ids=ids['val'],
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):

    files, ids = get_paths(opt.data_path, data_name, opt.use_restval)

    transform = get_transform(data_name, split_name, opt)
    """
    test_loader = get_loader_single(opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    vocab, transform, ids=ids[split_name],
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn)
    """
    test_loader = get_loader_single(opt.data_name, split_name,
                                   files[split_name],
                                   vocab, transform, ids=ids[split_name],
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn)

    return test_loader
