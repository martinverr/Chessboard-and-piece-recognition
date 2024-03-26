
# import torch 
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno
import torch
import torch.utils.data
from PIL import Image
import warnings
import utils

# import utils
import os, glob
import shutil
import numpy as np
import json

## from repo of tutorial of pytorch https://github.com/pytorch/vision/blob/main/references/segmentation
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat

def split_data(source_dir,base_name):
    """Split the data in two directory, source_dir + _data and source_dir + _annotation.
      The split is done from the mediatype of the file. If the file is .png it goes in the _data dir, if .json it goes to the _annotation.

    Args:
        source_dir (string): path of the directory with the data to split
        base_name (string): basename of the new directories

    Returns:
        _type_: _description_
    """
    # Create destination directories if they don't exist
    data_dir = os.path.join(source_dir, base_name + "_data")
    annotation_dir = os.path.join(source_dir, base_name + "_annotation")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    # Get list of files in source directory
    files = os.listdir(source_dir)

    # Iterate through files
    for file in files:
        # Check if file is a PNG image
        if file.endswith(".png"):
            # Move image file to data directory
            shutil.move(os.path.join(source_dir, file), os.path.join(data_dir, file))
        # Check if file is a JSON annotation file
        elif file.endswith(".json"):
            # Move annotation file to annotation directory
            shutil.move(os.path.join(source_dir, file), os.path.join(annotation_dir, file))
    return 0


#retrival bounding box and class from json file
def bb_mask_dict_annotation(annotation_path) -> list:
    """Retrival list of dictionaries of class and bounding box.

    Args:
        annotation_path (string): _description_

    Returns:
        list: [[{"class_label": lable , "bbox: [ xmin, ymin, xmax, ymax]"}]]
    """
    final_list = []
    # List all files in the directory
    files = os.listdir(annotation_path)

    # Filter JSON files
    json_files = [file for file in files if file.endswith('.json')]

    # load json file
    for file in json_files:
        class_lables = []
        coord_boxes = []
        boxes = []
        masks = []
        with open(os.path.join(annotation_path, file), 'r') as f:
            data_json = json.load(f)
            # Retrieve values for "piece" and "box" - poggers
            for piece in data_json["pieces"]:
                class_lables.append(torch.as_tensor(lable_to_number_conversion(piece["piece"]), dtype=torch.int64))
                coord_box = bb_conversion_coco(piece["box"])
                coord_boxes.append(coord_box)
                boxes.append(torch.as_tensor(coord_box, dtype =torch.float32))
                masks.append(torch.as_tensor(bb_to_mask_conversion(coord_box), dtype=torch.uint8))
            # annotations = [{"class_label": torch.as_tensor(lable_to_number_conversion(piece["piece"]),dtype=torch.int64), "bbox": torch.as_tensor(bb_conversion_coco(piece["box"]), dtype =torch.float32), "mask": bb_to_mask_conversion(piece["box"])} for piece in data_json["pieces"]]
        final_list.append({"labels": class_lables, "boxes": boxes, "masks": masks, "area_box": coord_boxes })
        # final_list.append(annotations)
    return final_list

def bb_to_mask_conversion(box, H = 800, W = 1200):
    """Given a box it return the mask (rectangular mask)

    Args:
        box (_type_): 4 point xmin,xmax,ymin,ymax
        H (int, optional): hight of the img Defaults to 800.
        W (int, optional): width (?) of the img Defaults to 1200.
    """
     # Initialize empty mask tensor
    mask = torch.zeros((H, W), dtype=torch.uint8)
    # TODO: change bb_mask_dict_annotation for more efficient box making (not using two times bb_conversion_coco)
    xmin, ymin, xmax, ymax = box
    mask[ymin:ymax, xmin:xmax] = 1
    return mask

def bb_conversion_coco(box):
    """Conversion from x,y,w,h to xmin,ymin,xmax,ymax.

    Args:
        box (list): list of 4 point to convert

    Returns:
        _type_: _description_
    """
    x,y,w,h = box
    xmin = int(x)
    ymin = int(y)
    xmax = int(x+w)
    ymax = int(y+h)
    # If ymax < ymin or xmax < xmin something went wrong
    if (ymax < ymin) or (xmax < xmin): print("Or ymax < ymin or xmax < xmin")
    return [xmin, ymin,xmax,ymax]

def lable_to_number_conversion(lable):

    if (lable == 'k'):
        return 1
    elif(lable == 'q'):
        return 2
    elif(lable == 'b'):
        return 3
    elif(lable == 'r'):
        return 4
    elif(lable == 'n'):
        return 5
    elif(lable == 'p'):
        return 6
    elif(lable == 'K'):
        return 7
    elif(lable == 'Q'):
        return 8
    elif(lable == 'B'):
        return 9
    elif(lable == 'R'):
        return 10
    elif(lable == 'N'):
        return 11
    elif(lable == 'P'):
        return 12
    
    raise TypeError("Illigal lable -  bombastic side-eye")



# Custom Dataset extended from torch.utils
class ChessboardDataset(Dataset):

    def __init__(self, data_path, annotations, transform=None):
        """Constructor for ChessboardDataset custom class son of torch.utils.data.Dataset. It needs the path data and the list of dictionaries of annotation.

        Args:
            data (_type_): path of the data
            annotations (_type_): dictionary of the annotation
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.data_path = data_path
        self.annotations = annotations
        self.transform = transform
        self.img_path = []

        # find img in the dir data_path
        for img_file in os.listdir(data_path):
            img_path = os.path.join(data_path, img_file)
            self.img_path.append(img_path)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Open img at idx index
        image_path = self.img_path[idx]
        image = Image.open(image_path).convert("RGB")

        # Add trasformation if needed
        if self.transform:
            image = self.transform(image)

        # Extract bounding box coordinates, class labels and mask from dict
            '''final_list.append({"labels": class_lables, "boxes": boxes, "masks": masks})'''
        annotation = self.annotations[idx]
        num_objs = len(annotation["labels"])
        boxes = torch.as_tensor(annotation["area_box"])
        target = {}
        target["boxes"] = annotation["boxes"]
        target["labels"] = annotation["labels"]
        target["masks"] = annotation["masks"]
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] =  torch.zeros((num_objs,), dtype=torch.int64)

        return image, target

# Class of the Mask Rcnn model
class MaskRcnn(nn.Module):
    """MaskRcnn class module imported from torch

    Args:
        nn (_type_): _description_
    """

    def __init__(self, number_classes, hidden_layer=256):
        super().__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, number_classes)

    def forward(self, x):
        return self.model(x)


def main():
    ## initilize utils variables
    base_dir = "dataset"
    source_directory_train = "train"
    source_directory_test = "test"
    source_directory_val = "val"
    dir_annotation ="_annotation"
    dir_data = "_data"

    ## variables for the model
    batch_size = 2
    num_epochs = 2

    # dividing dataset in different directories
    skip = False
    if skip:
        print("Processing dividing the dataset in train, test and val")
        # Split train
        split_data(os.path.join(base_dir,source_directory_train), source_directory_train)
        # Split test
        split_data(os.path.join(base_dir,source_directory_test), source_directory_test)
        # Split val
        split_data(os.path.join(base_dir,source_directory_val), source_directory_val)
    # bounding box retrival for every piece in every json file in every sub-dataset
    train_targets = bb_mask_dict_annotation(os.path.join(base_dir, source_directory_train, str(source_directory_train+dir_annotation)))
    test_targets = bb_mask_dict_annotation(os.path.join(base_dir, source_directory_test, str(source_directory_test+dir_annotation)))
    val_targets = bb_mask_dict_annotation(os.path.join(base_dir, source_directory_val, str(source_directory_val+dir_annotation)))

    # initilize dataset
    val_ds = ChessboardDataset(os.path.join(base_dir, source_directory_val,str(source_directory_val+dir_data)),val_targets)
    train_ds = ChessboardDataset(os.path.join(base_dir, source_directory_train,str(source_directory_train+dir_data)),train_targets)
    test_ds = ChessboardDataset(os.path.join(base_dir, source_directory_test,str(source_directory_test+dir_data)),test_targets)

    # initilize dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    validation_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html - non farina del mio sacco e non ci sto capendo un tubo
    # Crete Model
    model_ft = MaskRcnn(12)
    model_ft.model.to(device)
    # Define model parameters
    params = [p for p in model_ft.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.0005,
                                momentum=0.9,
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    for epoch in range(num_epochs):
        print(f"Epoche: {num_epochs}")
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model_ft.model, optimizer, validation_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model_ft, validation_dataloader, device=device)

    
    




if __name__ == "__main__":
    main()
