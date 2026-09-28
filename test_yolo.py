from ultralytics import YOLO

model = YOLO("yolo26m.pt")

results = model(
    r"C:\Users\Jayachandran\ProjectsAndDocs\Projects\Unityworks_vision_AI\vision-os-data\candidates\p9-live\v1\live-20260827-a\cam-13\cam-13_092955_860.jpg"
)

for result in results:
    print(result)