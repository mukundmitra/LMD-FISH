import torch
from ultralytics import SAM
from groundingdino.util.inference import load_model
from config import *
from utils.vision_utils import capture_and_detect, get_smallest_box
from utils.homography_utils import *
from utils.robot_utils import XArmController
from utils.ollama_utils import interpret_task, plan_actions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    sam = SAM(SAM_WEIGHTS)
    robot = XArmController()

    user_input = input("Describe your tool: ")
    location = interpret_task(user_input, model='llama3.1:8b-instruct-fp16')
    print(f"Relevant storage: {location}")

    boxes, shape = capture_and_detect(VIDEO_DEVICE_MAIN, model, TEXT_PROMPT1, BOX_THRESHOLD, TEXT_THRESHOLD)
    if boxes is None:
        print("No detection.")
        return

    box = get_smallest_box(boxes, shape[1], shape[0])
    pixel = [(box[0], box[1])]
    homography = cv2.findHomography(
        np.array([[0, 0], [0, shape[0]-1], [shape[1]-1, shape[0]-1], [shape[1]-1, 0]], dtype=np.float32),
        np.array([[0, 0], [18, 87], [77, 87], [90, 0]], dtype=np.float32)
    )[0]
    transformed = apply_homography(np.array(pixel), homography)
    real_coord = pixel_to_real(*transformed[0])
    coord = {
        'x': real_coord[0] * 10,
        'y': real_coord[1] * 10,
        'z': 199.8,
        'roll': -180,
        'pitch': 0,
        'yaw': 0
    }
    task = "Move an object from A to B."
    steps = plan_actions(task, model='llama3.1:8b-instruct-fp16')
    if steps and 'picked' in steps:
        status = robot.pickup(coord)
        print(f"Robot action: {status}")

if __name__ == "__main__":
    main()
