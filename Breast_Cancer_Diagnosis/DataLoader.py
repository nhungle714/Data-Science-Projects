class MamogramDataset(Dataset):

    def __init__(self, csv_file, root_dir, image_column, transform=None):
        """
        Args:
            csv_file (string): Csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column (string): name of the column image used
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_column = image_column
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.loc[idx, image_column])

        #image = io.imread(img_name,as_gray=True)
        image = pydicom.dcmread(image_name).pixel_array
        
        image = (image - image.mean()) / image.std()
            
        image_class = self.data_frame.loc[idx, 'class']

        sample = {'x': image[None,:], 'y': image_class}

        if self.transform:
            sample = self.transform(sample)

        return sample

def GetDataLoader(train_csv, validation_csv, test_csv, root_dir, image_column,
               train_transform, validation_transform, 
               batch_size, shuffle, num_workers): 
    Mamogram_TrainData = MamogramDataset(csv_file = train_csv,
                                        root_dir=root_dir, 
                                        image_column = image_column,
                                        transform=train_transform)
    train_loader = DataLoader(Mamogram_TrainData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)

    Mamogram_ValidationData = MamogramDataset(csv_file = validation_csv,
                                        root_dir=root_dir, 
                                        image_column = image_column,
                                        transform=train_transform)
    validation_loader = DataLoader(Mamogram_ValidationData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)


    Mamogram_TestData = MamogramDataset(csv_file = test_csv,
                                        root_dir=root_dir, 
                                        image_column = image_column,
                                        transform=None)
    test_loader = DataLoader(Mamogram_TestData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)
    
    dataset_sizes = {'train': len(Mamogram_TrainData), 
                     'val': len(Mamogram_ValidationData)}
    return train_loader, validation_loader, test_loader, dataset_sizes





class MamogramDataset_TL(Dataset):

    def __init__(self, csv_file, root_dir, image_column, num_channel, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column (string): name of the column image used
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_column = image_column
        self.num_channel = num_channel
        self.samples = []
        
        for idx in range(len(self.data_frame)):
            img_name = os.path.join(self.root_dir,
                                    self.data_frame.loc[idx, image_column])

            image = pydicom.dcmread(image_name).pixel_array
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = image[:,:,0]

            image=np.repeat(image[None,...], num_channel, axis=0)

            image_class = self.data_frame.loc[idx, 'class']

            if self.transform:
                image = self.transform(image)

            sample = {'x': image, 'y': image_class}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.samples[idx]


def GetDataLoader_TL(train_csv, validation_csv, test_csv, root_dir, image_column,
               train_transform, validation_transform, num_channel,
               batch_size, shuffle, num_workers): 
    Mamogram_TrainData = MamogramDataset_TL(csv_file = train_csv,
                                        root_dir=root_dir, 
                                        image_column = image_column,
                                        num_channel = num_channel,
                                        transform=train_transform)
    train_loader = DataLoader(Mamogram_TrainData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)

    Mamogram_ValidationData = MamogramDataset_TL(csv_file = validation_csv,
                                        root_dir=root_dir, 
                                        image_column = image_column,
                                        num_channel = num_channel,
                                        transform=train_transform)
    validation_loader = DataLoader(Mamogram_ValidationData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)


    Mamogram_TestData = MamogramDataset_TL(csv_file = test_csv,
                                        root_dir=root_dir, 
                                        image_column = image_column,
                                        num_channel = num_channel,
                                        transform=None)
    test_loader = DataLoader(Mamogram_TestData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)
    
    dataset_sizes = {'train': len(Mamogram_TrainData), 
                     'val': len(Mamogram_ValidationData)}
    return train_loader, validation_loader, test_loader, dataset_sizes

