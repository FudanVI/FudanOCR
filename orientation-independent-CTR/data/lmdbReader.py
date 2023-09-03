import torch
import torchvision.transforms as transforms
import lmdb
import six
import sys

from PIL import Image
from torch.utils.data import Dataset

class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, reverse=False, alphabet=None):

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error 报错index为 %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')

                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            label = strQ2B(label)
            label += '$'
            label = label.lower()


            if self.transform is not None:
                img = self.transform(img)
        return (img, label, index)

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.width, self.height = self.size

    def __call__(self, img, to_vertical, is_vertical):
        if not is_vertical:
            if not to_vertical:
                img = img.resize((self.width, self.height), self.interpolation)
            else:
                img = img.transpose(Image.ROTATE_270)
                img = img.resize((self.height, self.width), self.interpolation)
        else:
            if not to_vertical:
                img = img.resize((self.height, self.width), self.interpolation)
            else:
                img = img.transpose(Image.ROTATE_90)
                img = img.resize((self.width, self.height), self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images, labels, _ = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        raw_img = images

        transform = resizeNormalize((imgW, imgH))
        images, images_v, is_v = [], [], []

        for img in raw_img:
            w, h = img.size
            if 1.5 * w >= h:
                is_v.append(0)
                images.append(transform(img, to_vertical=False, is_vertical=False))
                images_v.append(transform(img, to_vertical=True, is_vertical=False))
            else:
                is_v.append(1)
                images.append(transform(img, is_vertical=True, to_vertical=True))
                images_v.append(transform(img, is_vertical=True, to_vertical=False))

        images = torch.cat([item.unsqueeze(0) for item in images], 0)
        images_v = torch.cat([item.unsqueeze(0) for item in images_v], 0)

        return images, images_v, labels, is_v


