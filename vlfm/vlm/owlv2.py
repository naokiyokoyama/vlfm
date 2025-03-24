from typing import List

import numpy as np
import torch
from vlfm.vlm.detections import ObjectDetections
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image, CUDALockMixin


class OWLv2:
    def __init__(self):
        """Loads the model and saves it to a field."""
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)

    def predict(self, image: np.ndarray, classes: List[str]) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            classes (List[str]): A list of classes to detect.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        print("Candidate classes:", classes)
        pil_image = Image.fromarray(image)
        inputs = self.processor(text=[classes], images=pil_image, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.inference(**inputs)

        h, w = inputs.pixel_values.shape[-2:]

        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.Tensor([(h, w)]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.2
        )
        boxes, logits, labels = [results[0][i] for i in ["boxes", "scores", "labels"]]
        phrases = []
        for box, score, label in zip(boxes, logits, labels):
            phrases.append(classes[label])
            print(
                f"Detected {classes[label]} with confidence {round(score.item(), 3)} at"
                f" location {[round(i, 2) for i in box.tolist()]}"
            )
        orig_h, orig_w = image.shape[:2]
        if orig_w > orig_h:
            new_h = orig_h * w / orig_w
            norm_bboxes = boxes.cpu() / torch.Tensor([w, new_h, w, new_h])
        else:
            new_w = orig_w * h / orig_h
            norm_bboxes = boxes.cpu() / torch.Tensor([new_w, h, new_w, h])
        detections = ObjectDetections(
            norm_bboxes, logits, phrases, image_source=image, fmt="xyxy"
        )

        return detections

    def inference(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model(*args, **kwargs)

class OWLv2Client:
    def __init__(self, port: int = 12186):
        self.url = f"http://localhost:{port}/owlv2"

    def predict(self, image_numpy: np.ndarray, classes: List[str]) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, classes=classes)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading model...")

    class OWLv2Server(CUDALockMixin, ServerMixin, OWLv2):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload.pop("image"))
            return self.predict(image, **payload).to_json()

    owlv2 = OWLv2Server()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(owlv2, name="owlv2", port=args.port)

    # The commented code below can be used for testing without hosting
    # o = OWLv2()
    # import cv2
    # print("Model loaded!")
    # img_bgr = cv2.imread(
    #     "/coc/testnvme/nyokoyama3/spring2024/vlfm_ovon/people_and_bus.jpg"
    # )
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # detections = o.predict(img_rgb, "person . bus .")
    # output_path = "output_detection.jpg"
    # cv2.imwrite(
    #     output_path, cv2.cvtColor(detections.annotated_frame, cv2.COLOR_RGB2BGR)
    # )
    # print(f"Annotated image saved to {output_path}")
