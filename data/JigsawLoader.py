import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class JigsawDataset(data.Dataset):
    def __init__(self, data_path, txt_list, classes=100, patches=True):
        self.data_path = data_path
        self.names, self.labels = self.__dataset_info(txt_list)
        self.N = len(self.names)
        self.permutations = self.__retrieve_permutations(classes)
        self.grid_size = 3

        self._image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)]
        )
        self._augment_tile = transforms.Compose([
            transforms.RandomCrop(68),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomGrayscale(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if patches:
            self.returnFunc = lambda x: x
        else:

            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)

        self.returnFunc = make_grid

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(n_grids)]
        data = torch.stack(data, 0)
        return self.returnFunc(data), int(order), int(self.labels[index])

    def __len__(self):
        return len(self.names)

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm
    
    
class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._augment_tile = transforms.Compose([
#             transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile

        data = torch.stack(tiles, 0)
        return self.returnFunc(data), 0, int(self.labels[index])
