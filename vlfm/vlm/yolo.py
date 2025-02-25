import numpy as np
import torch
import ultralytics

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image


class YOLO:
    def __init__(self):
        """Loads the model and saves it to a field."""
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = ultralytics.YOLO("data/yolo11x.pt").to(self.device)

    def predict(self, image: np.ndarray) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """
        # Expects BGR images when in numpy format
        # https://docs.ultralytics.com/modes/predict/#inference-sources
        image_bgr = image[:, :, ::-1]
        with torch.inference_mode():  # Calculating gradients causes a GPU memory leak
            result = self.model(image_bgr)[0]
        # Save the image
        detections = results_to_object_detections(result)
        return detections


class YOLOClient:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/yolo"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


def results_to_object_detections(
    results: ultralytics.engine.results.Results,
) -> ObjectDetections:
    boxes = results.boxes.xyxy
    logits = results.boxes.conf
    class_labels = results.names

    # Get the image source from the results
    image_source = results.orig_img

    # Convert class labels to phrases
    phrases = [class_labels[int(label)] for label in results.boxes.cls]

    boxes = normalize_bbox_coordinates(
        boxes, image_source.shape[0], image_source.shape[1]
    )

    # Create an instance of ObjectDetections
    object_detections = ObjectDetections(
        boxes=boxes,
        logits=logits,
        phrases=phrases,
        image_source=image_source,
        fmt="xyxy",
    )

    return object_detections


def normalize_bbox_coordinates(bbox_tensor, height, width):
    """
    Normalize the bounding box coordinates by the given height and width.

    Args:
        bbox_tensor (torch.Tensor): Tensor of shape (N, 4) containing bounding box
                                    coordinates in xyxy format.
        height (float): Height of the image.
        width (float): Width of the image.

    Returns:
        torch.Tensor: Tensor of shape (N, 4) with normalized bounding box coordinates.
    """
    # Clone the input tensor to avoid modifying the original
    normalized_tensor = bbox_tensor.clone()

    # Normalize the x coordinates by the width
    normalized_tensor[:, 0] /= width
    normalized_tensor[:, 2] /= width

    # Normalize the y coordinates by the height
    normalized_tensor[:, 1] /= height
    normalized_tensor[:, 3] /= height

    return normalized_tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    # Add optional bool flag called test
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("Loading model...")

    if args.test:
        # Load the model
        import cv2

        yolo = YOLO()
        print("Model loaded!")
        print("Testing model...")
        img_path = "data/bus.jpg"
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        od = yolo.predict(img_rgb)
        ann_img = od.annotated_frame
        cv2.imwrite("annotated_frame.jpg", cv2.cvtColor(ann_img, cv2.COLOR_RGB2BGR))
        quit()

    class YOLOServer(ServerMixin, YOLO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    yolo = YOLOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolo, name="yolo", port=args.port)
