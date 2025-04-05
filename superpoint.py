from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load two images
url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
image_2 = Image.open(requests.get(url_image_2, stream=True).raw)

images = [image_1, image_2]

# Load processor and model
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

# Inference
inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Postprocess
image_sizes = [(image.height, image.width) for image in images]
outputs = processor.post_process_keypoint_detection(outputs, image_sizes)

# Visualize keypoints on image 1
plt.axis("off")
plt.imshow(image_1)

keypoints = outputs[0]["keypoints"].detach().cpu().numpy()
scores = outputs[0]["scores"].detach().cpu().numpy()

plt.scatter(
    keypoints[:, 0],
    keypoints[:, 1],
    c=scores * 100,
    s=scores * 50,
    alpha=0.8
)
plt.title("SuperPoint Keypoints - Image 1")
plt.savefig("images/kitty_superoint")
plt.show()
