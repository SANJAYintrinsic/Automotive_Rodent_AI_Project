"""

!pip install ultralytics


from ultralytics import YOLO
YOLO("yolov8n.pt")


from google.colab import files
uploaded = files.upload()


!unzip Rodent.v2i.yolov8.zip


!ls


import yaml

data_yaml = "/content/data.yaml"

with open(data_yaml, 'r') as f:
    data = yaml.safe_load(f)

data['path'] = "/content"

with open(data_yaml, 'w') as f:
    yaml.dump(data, f)

print("data.yaml fixed ")


from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano model = fast + lightweight

model.train(
    data="/content/data.yaml",
    epochs=80,
    imgsz=416,
    batch=16,
    device=0,
    name="rodent_detector"
)


from IPython.display import Image, display
display(Image(filename="/content/runs/detect/rodent_detector3/results.png"))


from google.colab import files
files.download("/content/runs/detect/rodent_detector3/weights/best.pt")



from ultralytics import YOLO

model = YOLO("/content/runs/detect/rodent_detector3/weights/best.pt")

model.predict(
    source="/content/test/images/R-20-_png.rf.aba56765018e1870056e1a0790241ba9.jpg",
    conf=0.5,
    save=True
)


!ls runs/detect/predict


from IPython.display import Image, display
import os

pred_path = "/content/runs/detect/predict"

for img in os.listdir(pred_path):
    display(Image(filename=f"{pred_path}/{img}"))


from google.colab import files
uploaded = files.upload()


from ultralytics import YOLO

model = YOLO("/content/runs/detect/rodent_detector3/weights/best.pt")

model.predict(
    source="Enhanced_Rat_Detection_Simulation.mp4",
    conf=0.5,
    save=True
)

"""
