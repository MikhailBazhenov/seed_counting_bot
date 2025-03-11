# https://t.me/SeedsCountingBot
# pip install ultralytics
# pip install python-telegram-bot
# pip install Pillow
# pip install torchvision


from typing import Final
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes
import io
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

TOKEN: Final = "0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
BOT_USERNAME: Final = "@SeedsCountingBot"
counter: int = 0

# load the model
model = YOLO(r'yolo11_seeds.pt')


# regular boxes
def soft_nms(boxes, sigma=0.5, Nt=0.3, method=2, score_thresh=0.001):
    """
    Applies Soft Non-Maximum Suppression (Soft-NMS) to remove excessive overlapping boxes.
    
    Parameters:
        boxes (list of lists): List of bounding boxes in the format [x1, y1, x2, y2, confidence, class_id].
        sigma (float): Controls the suppression factor in Gaussian decay (default: 0.5).
        Nt (float): IoU threshold for suppression (default: 0.3).
        method (int): 
            0 = Original NMS (hard threshold)
            1 = Linear decay
            2 = Gaussian decay (default)
        score_thresh (float): Threshold below which boxes are removed.
    
    Returns:
        list: Filtered boxes after applying Soft-NMS.
    """
    boxes = np.array(boxes)
    N = len(boxes)
    
    for i in range(N):
        max_pos = i
        max_score = boxes[i, 4]

        # Find the highest confidence box
        for j in range(i + 1, N):
            if boxes[j, 4] > max_score:
                max_score = boxes[j, 4]
                max_pos = j

        # Swap the highest confidence box to the front
        boxes[i], boxes[max_pos] = boxes[max_pos], boxes[i]

        x1_i, y1_i, x2_i, y2_i, score_i, class_id_i = boxes[i]

        # Iterate through the remaining boxes
        for j in range(i + 1, N):
            x1_j, y1_j, x2_j, y2_j, score_j, class_id_j = boxes[j]

            # Compute IoU
            xx1 = max(x1_i, x1_j)
            yy1 = max(y1_i, y1_j)
            xx2 = min(x2_i, x2_j)
            yy2 = min(y2_i, y2_j)

            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            union = ((x2_i - x1_i + 1) * (y2_i - y1_i + 1)) + ((x2_j - x1_j + 1) * (y2_j - y1_j + 1)) - inter
            IoU = inter / union if union > 0 else 0

            # Apply Soft-NMS decay based on the chosen method
            if method == 1:  # Linear
                if IoU > Nt:
                    boxes[j, 4] *= (1 - IoU)
            elif method == 2:  # Gaussian
                boxes[j, 4] *= np.exp(-(IoU**2) / sigma)
            else:  # Standard NMS
                if IoU > Nt:
                    boxes[j, 4] = 0  # Directly suppress

    # Filter out boxes with low scores
    filtered_boxes = [box.tolist() for box in boxes if box[4] > score_thresh]
    return filtered_boxes


# normalization of image contrast and rotation
def image_normalization(img):
    # image normalization
    maximum = np.percentile(img.mean(axis=2), 99)
    minimum = np.percentile(img.mean(axis=2), 1)
    alpha = 255 / (maximum - minimum)
    beta = - minimum * alpha
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    rotated = False

    height, width, channels = img.shape

    # rotate if not vertical image
    if height < width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height
        rotated = True

    desired_image_size = (1280, 960)
    original_aspect_ratio = height / width
    desired_aspect_ratio = desired_image_size[0] / desired_image_size[1]

    # resize image if necessary keeping aspect ratio
    if desired_image_size != (height, width):

        if original_aspect_ratio == desired_aspect_ratio:
            img = cv2.resize(img, (desired_image_size[1], desired_image_size[0]))

        elif original_aspect_ratio > desired_aspect_ratio:
            img = cv2.resize(img, (int(desired_image_size[0] / original_aspect_ratio), desired_image_size[0]))
            image_extended = np.ndarray((desired_image_size[0], desired_image_size[1], channels), dtype=img.dtype)
            image_extended.fill(255)
            image_extended[:, int((desired_image_size[1] - img.shape[1])/2):int((desired_image_size[1] - img.shape[1])/2) + img.shape[1], :] = img
            img = image_extended

        elif original_aspect_ratio < desired_aspect_ratio:
            img = cv2.resize(img, (desired_image_size[1], int(desired_image_size[1] * original_aspect_ratio)))
            image_extended = np.ndarray((desired_image_size[0], desired_image_size[1], channels), dtype=img.dtype)
            image_extended.fill(255)
            image_extended[int((desired_image_size[0] - img.shape[0])/2):int((desired_image_size[0] - img.shape[0])/2) + img.shape[0], :, :] = img
            img = image_extended
    
    return(img, rotated)


# regular boxes
def run_yolo_inference(image_path, score_thresh=0.2):
    """
    Runs YOLO object detection on an image and applies Soft-NMS.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Apply autocontrast
    img, rotated = image_normalization(img=img)
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    # Run YOLO inference
    results = model(img)
    results_rotated = model(img_rotated)
    
    detections = results[0].cpu().boxes.data.numpy()  # Get detected boxes (x1, y1, x2, y2, confidence, class)
    detections_rotated = results_rotated[0].cpu().boxes.data.numpy()

    height, width, channels = img.shape
    # Convert YOLO output to Soft-NMS format
    yolo_boxes = [[x1, y1, x2, y2, conf, cls] for x1, y1, x2, y2, conf, cls in detections]
    yolo_boxes_rotated = [[y1, height - x2, y2, height - x1, conf, cls] for x1, y1, x2, y2, conf, cls in detections_rotated]
    yolo_boxes = yolo_boxes + yolo_boxes_rotated

    # Apply Soft-NMS
    filtered_boxes = soft_nms(yolo_boxes, sigma=0.5, Nt=0.3, method=2, score_thresh=score_thresh)
    count = np.zeros(len(model.names))

    # Draw filtered boxes
    for x1, y1, x2, y2, conf, cls in filtered_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # label = f"{model.names[int(cls)]}: {conf:.2f}"
        label = str(int(cls))
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if cls == 0:
            cv2.circle(img, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)
        elif cls == 1:
            cv2.circle(img, (center_x, center_y), radius=5, color=(241, 181, 62), thickness=-1)
        count[int(cls)] += 1

    # Show result
    cv2.imwrite(image_path[:-4] + '_processed.jpg', img)
    return(count)



# Define a few command handlers. These usually take the two arguments update and context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Hello, {update.effective_user.first_name}!")
    await update.message.reply_text("Send me a photo and I'll count the sunflower seeds on it..")


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("I can count sunflower seeds..")
    await update.message.reply_text("Place the sunflower seeds individually on a white sheet of paper, take a picture and send it to me.")

# Response to a photo
async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global counter
    counter += 1
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    
    # Save the photo to a temporary file
    with io.BytesIO(photo_bytes) as file:
        file_name = f'photo_{counter:09d}.jpg'
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "wb") as f:
            f.write(file.read())

    # Do something with the photo
    counts = run_yolo_inference(file_path, score_thresh=0.25)

    await update.message.reply_photo(file_name[:-4] + '_processed.jpg')

    await update.message.reply_text(f'Normal seeds: {int(counts[0])}; sterile seeds: {int(counts[1])}')


def handle_response(text: str) -> str:
    if 'hello' in text.lower():
        return "Hello!"
    elif 'bye' in text.lower():
        return "Bye!"
    elif 'how are you' in text.lower():
        return "Good!"
    elif 'what is your name' in text.lower():
        return "My name is " + BOT_USERNAME
    elif 'what can you do' in text.lower():
        return "I can count sunflower seeds."
    elif 'who are you' in text.lower():
        return "My name is " + BOT_USERNAME
    else:
        return "I can count sunflower seeds, and I can't answer other questions."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_type = update.message.chat.type
    text = update.message.text
    print(f"User {update.message.chat.id} in {message_type} sent {text}")
    if message_type == "group":
        if BOT_USERNAME in text:
            text = text.replace(BOT_USERNAME, "").strip()
            response = handle_response(text)
            await update.message.reply_text(response)
        await update.message.reply_text("I'm sorry, but I can't answer group messages.")
    elif message_type == "private":
        response = handle_response(text)
        await update.message.reply_text(response)
    print(f'Response: {response}')

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()

    start_handler = CommandHandler('start', start)
    help_handler = CommandHandler('help', help)
    photo_handler = MessageHandler(filters.PHOTO, photo)
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)

    app.add_handler(start_handler)
    app.add_handler(help_handler)
    app.add_handler(photo_handler)
    app.add_handler(message_handler)

    app.run_polling(poll_interval=1)

    # Run the bot