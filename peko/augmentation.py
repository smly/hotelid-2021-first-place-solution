import albumentations as albu


def get_augv5(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(input_size, input_size, scale=(0.6, 1.0), p=1.0),
        albu.Resize(width=input_size, height=input_size),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=1),
            albu.RandomContrast(0.1, p=1),
            albu.RandomGamma(p=1)
        ], p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
        albu.Cutout(p=0.5, max_h_size=80, max_w_size=80, num_holes=2),
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
    ])
    val_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
    ])
    return train_transform, val_transform


def get_augv5_no_flip(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(input_size, input_size, scale=(0.6, 1.0), p=1.0),
        albu.Resize(width=input_size, height=input_size),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=1),
            albu.RandomContrast(0.1, p=1),
            albu.RandomGamma(p=1)
        ], p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
        albu.Cutout(p=0.5, max_h_size=80, max_w_size=80, num_holes=2),
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
    ])
    val_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
    ])
    return train_transform, val_transform
