from gradcam import make_gradcam
import cv2


image_path = "D:\\ML projects\\ML Leaf disease\\data\\raw\\diseased\\diseased 5.webp"

result = make_gradcam(image_path)

cv2.imwrite(
    "gradcam_result.jpg",
    cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
)

print("Grad-CAM generated successfully!")
print("Saved as: gradcam_result.jpg")