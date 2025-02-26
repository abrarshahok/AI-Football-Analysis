from ultralytics import YOLO

model = YOLO('./models/best.pt')

input_video_path = './input/input.mp4'

results = model.predict(input_video_path, save=True)
print(results[0])

print('===========================================')

for box in results[0].boxes:
    print(box)