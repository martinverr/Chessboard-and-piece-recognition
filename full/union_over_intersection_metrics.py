from generate_bboxgt import from_json_to_annotation
import cv2
import torch
import numpy as np
import glob
import torchvision

'''IoU is a crucial metric for assessing segmentation models, commonly called Jaccard's Index,
 since it quantifies how well the model can distinguish objects from their backgrounds in an image.'''

def calculate_iou(mask_one:torch.Tensor, mask_two: torch.Tensor ) -> torch.Tensor :
    intersection = torch.logical_and(mask_one,mask_two)
    union = torch.logical_or(mask_one,mask_two)
    #np.sum(intersection) / np.sum(union)
    iou = torch.sum(intersection)/torch.sum(union)
    return iou


def main():
    file_path_annotation = 'json_annotation_gt_pieces.json'
    file_path_annotation_custom =  'json_annotation_custom_pieces.json'
    input_imgs = glob.glob('./input/**.png')
    input_imgs = [x.split("\\")[-1].strip('.png') for x in input_imgs]

    gt_bbox_list = []
    custom_bbox_list = []
    tp_counts = {
                'w_King': 0, 'w_Queen': 0, 'w_Rook' : 0,  'w_Bishop': 0, 'w_Knight': 0, 'w_Pawn': 0,
                'b_King':0, 'b_Queen' : 0, 'b_Rook': 0, 'b_Bishop': 0, 'b_Knight': 0, 'b_Pawn': 0
            }
    fp_counts =  {
                'w_King': 0, 'w_Queen': 0, 'w_Rook' : 0,  'w_Bishop': 0, 'w_Knight': 0, 'w_Pawn': 0,
                'b_King':0, 'b_Queen' : 0, 'b_Rook': 0, 'b_Bishop': 0, 'b_Knight': 0, 'b_Pawn': 0
            }
    fn_counts ={
                'w_King': 0, 'w_Queen': 0, 'w_Rook' : 0,  'w_Bishop': 0, 'w_Knight': 0, 'w_Pawn': 0,
                'b_King':0, 'b_Queen' : 0, 'b_Rook': 0, 'b_Bishop': 0, 'b_Knight': 0, 'b_Pawn': 0
            }
    
    iou_threshold = 0.5

    for name in input_imgs:
        # print(f"Processing img {name}")

        gt_dic = from_json_to_annotation(file_path_annotation, name)
        custom_dic = from_json_to_annotation(file_path_annotation_custom,name)
        if gt_dic and custom_dic:
            if len(gt_dic[f"{name}"]) == len(custom_dic[f"{name}"]):
                gt_dic = gt_dic[f"{name}"]
                custom_dic = custom_dic[f"{name}"]

                for gt in gt_dic:
                    cu = [c for c in custom_dic if c["position"] == gt["position"]]
                    cu = cu[0]
                    iou = calculate_iou(gt["mask"], cu["mask"])
                    if iou >= iou_threshold:
                        tp_counts[gt["piece"]] += 1
                    else:
                        fp_counts[cu["piece"]] += 1
                    

                #     # Calculate precision, recall, and F1 score for each class
                # precision = tp_counts / (tp_counts + fp_counts)
                # recall = tp_counts / (tp_counts + fn_counts)
                # f1_score = 2 * (precision * recall) / (precision + recall)


            else:
                print("Not equal number of pieces")
        else:
            print("Json not found or error in json indentation")

    # print(f"Quelle che si sovrappongono bene: {tp_counts} \nQuelle che si sovrappongono meno bene: {fp_counts}")
    for t_k, t_v in tp_counts.items():
        try:
            percent_segm = t_v/(fp_counts[t_k]+t_v)
            print(f"For class {t_k} the % of good segmentation in the entire dataset is: {percent_segm}")
        except TypeError as e:
            print(e)
        else:
            continue


def coco_to_pascal_voc(bbox):
    x1, y1, w, h = bbox
    return torch.Tensor([[x1,y1, x1 + w, y1 + h]])
        
def main_two():
    file_path_annotation = 'json_annotation_gt_pieces.json'
    file_path_annotation_custom =  'json_annotation_custom_pieces.json'
    input_imgs = glob.glob('./input/**.png')
    input_imgs = [x.split("\\")[-1].strip('.png') for x in input_imgs]
    iou_mean = {
                'w_King': [], 'w_Queen': [], 'w_Rook' : [],  'w_Bishop': [], 'w_Knight': [], 'w_Pawn': [],
                'b_King':[], 'b_Queen' : [], 'b_Rook': [], 'b_Bishop': [], 'b_Knight': [], 'b_Pawn': []
            }
    for name in input_imgs:

        gt_dic = from_json_to_annotation(file_path_annotation, name)
        custom_dic = from_json_to_annotation(file_path_annotation_custom,name)
        
        if gt_dic == None and custom_dic == None:
            print("Img in file annotation not found")
            continue
        
        if gt_dic and custom_dic:
            if len(gt_dic[f"{name}"]) == len(custom_dic[f"{name}"]):
                gt_dic = gt_dic[f"{name}"]
                custom_dic = custom_dic[f"{name}"]
                gt_dic = sorted(gt_dic, key=lambda x: x['position'])
                custom_dic = sorted(custom_dic, key=lambda x: x['position'])
                for gt, cu in zip(gt_dic,custom_dic):
                    if gt["position"] == cu["position"]:
                        iou = float(torchvision.ops.box_iou(coco_to_pascal_voc(gt['bbox']), coco_to_pascal_voc(cu['bbox'])))
                        iou_mean[gt['piece']].append(iou)
                    else:
                        raise Exception("Or the sort does not work or there is a problem with the annotation of the position")
            else:
                print("Not equal number of pieces")
        else:
            print("Json not found or error in json indentation")
    final_mean_iou = {k: np.mean(v) for k, v in iou_mean.items()}

    for k,v in final_mean_iou.items():
        print(f"Class: {k}\t|\tIOU value: {v}")





if __name__ == "__main__":
    main_two()