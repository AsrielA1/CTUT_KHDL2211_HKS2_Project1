import torch

def intersection_area(boxes1, boxes2):
  """
  Calculates the intersection area between two tensors of bounding boxes.

  Args:
      boxes1: A torch tensor of shape (batch_size, 4) representing bounding boxes 
              with format [x1, y1, x2, y2].
      boxes2: A torch tensor of the same shape as boxes1.

  Returns:
      A torch tensor of shape (batch_size,) representing the intersection area 
      for each bounding box in the batch.
  """
  # Get the top left corner coordinates for both boxes
  tl_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
  tl_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])

  # Get the bottom right corner coordinates for both boxes
  br_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
  br_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

  # Ensure width and height are non-negative (no intersection)
  intersection_width = torch.clamp(br_x2 - tl_x1, min=0)
  intersection_height = torch.clamp(br_y2 - tl_y1, min=0)

  # Calculate the intersection area
  intersection_area = intersection_width * intersection_height

  return intersection_area

# Example usage
boxes1 = torch.tensor([[1, 2, 4, 5], [3, 4, 6, 7]])
boxes2 = torch.tensor([[2, 3, 5, 6], [1, 1, 5, 5]])

intersection_areas = intersection_area(boxes1, boxes2)
print(intersection_areas)

boxes1 = torch.tensor([[0, 0, 1, 0]])
boxes2 = torch.tensor([[1, 0, 1, 1]])
intersection_areas = intersection_area(boxes1, boxes2)
print(intersection_areas)
