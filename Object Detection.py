from ultralytics import YOLO
import cv2

def load_model():
    return YOLO('yolov8n.pt')

def open_video(video_path):
    vid_captured = cv2.VideoCapture(video_path)
    if not vid_captured.isOpened():
        raise Exception("Error: Couldn't open video file or stream.")
    return vid_captured

def read_frame(vid_captured):
    ret, frame = vid_captured.read()
    if not ret:
        return None
    return frame

def resize_frame(frame, scale_percent=80):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def detect_and_plot(model, frame):
    results = model.track(frame, persist=True)
    return results[0].plot()

def display_frame(frame):
    cv2.imshow('YOLOv8 Object Detection', frame)

def process_video(video_path):
    vid_captured = open_video(video_path)
    model = load_model()

    while True:
        frame = read_frame(vid_captured)
        if frame is None:
            print("End of video or failed to grab frame.")
            break

        frame = detect_and_plot(model, frame)
        frame = resize_frame(frame)

        display_frame(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vid_captured.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r'D:\Matab Mohsen\codealpha\obj\Test Video.mp4'
    process_video(video_path)
