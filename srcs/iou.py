import torch

def intersection_over_union(
        box_preds: torch.Tensor,
        box_labels: torch.Tensor,
        box_format: str="corners"
) -> torch.Tensor:
    
    if box_format == "corners":
        box1_x1 = box_preds[:, 0]
        box1_y1 = box_preds[:, 1]
        box1_x2 = box_preds[:, 2]
        box1_y2 = box_preds[:, 3]

        box2_x1 = box_labels[:, 0]
        box2_y1 = box_labels[:, 1]
        box2_x2 = box_labels[:, 2]
        box2_y2 = box_labels[:, 3]

    elif box_format == "midpoint":
        box1_x1 = box_preds[:, 0] - box_preds[:, 2] / 2
        box1_y1 = box_preds[:, 1] - box_preds[:, 3] / 2
        box1_x2 = box_preds[:, 0] - box_preds[:, 2] / 2
        box1_y1 = box_preds[:, 1] - box_preds[:, 3] / 2

        box2_x1 = box_labels[:, 0] - box_labels[:, 2] / 2
        box2_y1 = box_labels[:, 1] - box_labels[:, 3] / 2
        box2_x2 = box_labels[:, 0] - box_labels[:, 2] / 2
        box2_y1 = box_labels[:, 1] - box_labels[:, 3] / 2

    else:
        raise ValueError("\nbox_format=[midpoint|corners]\n")
    
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection_width = torch.clamp(x2 - x1, min=0)
    intersection_height = torch.clamp(y2 - y1, min=0)

    intersection = intersection_height * intersection_width

    union = box1_area + box2_area - intersection + 1e-8    

    return intersection / union
