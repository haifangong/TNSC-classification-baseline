import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        image_basic = sample['image_basic']
        image_context = sample['image_context']
        image_margin = sample['image_margin']
        mask = sample['label']

        if self.padding > 0:
            image_basic = ImageOps.expand(image_basic, border=self.padding, fill=0)
            image_context = ImageOps.expand(image_context, border=self.padding, fill=0)
            image_margin = ImageOps.expand(image_margin, border=self.padding, fill=0)

        # assert img.size == mask.size
        w, h = image_basic.size
        th, tw = self.size  # target size
        if w == tw and h == th:
            return {'image_basic': image_basic,
                    'image_context': image_context,
                    'image_margin': image_margin,

                    'label': mask}
        if w < tw or h < th:
            image_basic = image_basic.resize((tw, th), Image.BILINEAR)
            image_context = image_context.resize((tw, th), Image.BILINEAR)
            image_margin = image_margin.resize((tw, th), Image.BILINEAR)

            return {'image_basic': image_basic,
                    'image_context': image_context,
                    'image_margin': image_margin,
                    'label': mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        image_basic = image_basic.crop((x1, y1, x1 + tw, y1 + th))
        image_context = image_context.crop((x1, y1, x1 + tw, y1 + th))
        image_margin = image_margin.crop((x1, y1, x1 + tw, y1 + th))

        return {'image_basic': image_basic,
                'image_context': image_context,
                'image_margin': image_margin,
                'label': mask}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        # mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image_basic = sample['image_basic']
        image_context = sample['image_context']
        image_margin = sample['image_margin']
        mask = sample['label']
        if random.random() < 0.5:
            image_basic = image_basic.transpose(Image.FLIP_LEFT_RIGHT)
            image_context = image_context.transpose(Image.FLIP_LEFT_RIGHT)
            image_margin = image_margin.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image_basic': image_basic,
                'image_context': image_context,
                'image_margin': image_margin,
                'label': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image_basic = np.array(sample['image_basic']).astype(np.float32)
        image_context = np.array(sample['image_context']).astype(np.float32)
        image_margin = np.array(sample['image_margin']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        image_basic /= 255.0
        image_basic -= self.mean
        image_basic /= self.std
        image_context /= 255.0
        image_context -= self.mean
        image_context /= self.std
        image_margin /= 255.0
        image_margin -= self.mean
        image_margin /= self.std

        return {
            'image_basic': image_basic,
            'image_context': image_context,
            'image_margin': image_margin,
            'mask': mask,
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_basic = np.array(sample['image_basic']).astype(np.float32).transpose((2, 0, 1))
        image_context = np.array(sample['image_context']).astype(np.float32).transpose((2, 0, 1))
        image_margin = np.array(sample['image_margin']).astype(np.float32).transpose((2, 0, 1))

        # img = np.expand_dims(np.array(sample['image']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))


        image_basic = torch.from_numpy(image_basic).float()
        image_context = torch.from_numpy(image_context).float()
        image_margin = torch.from_numpy(image_margin).float()

        mask = np.array(sample['mask']).astype(np.float32)
        mask = torch.from_numpy(mask).long()

        return {
            'image_basic': image_basic,
            'image_context': image_context,
            'image_margin': image_margin,
            'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        image_basic = sample['image_basic']
        image_context = sample['image_context']
        image_margin = sample['image_margin']

        mask = 0

        # assert img.size == mask.size

        image_basic = image_basic.resize(self.size, Image.BILINEAR)
        image_context = image_context.resize(self.size, Image.BILINEAR)
        image_margin = image_margin.resize(self.size, Image.BILINEAR)
        # mask = mask.resize(self.size, Image.NEAREST)

        return {
            'image_basic': image_basic,
            'image_context': image_context,
            'image_margin': image_margin,
            'label': mask}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        # mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomRotateOrthogonal(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.randint(0, 3) * 90
        if rotate_degree > 0:
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask}

        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}
