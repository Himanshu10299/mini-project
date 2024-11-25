# import torch
# import cv2

# # Vehicle classes in YOLOv5 (from COCO dataset)
# VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

# def process_image(image_path, model):
#     """
#     Processes an image, detects vehicles, displays the output, and saves the annotated image.
#     """
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Perform YOLOv5 inference
#     results = model(img)
    
#     # Extract detections
#     detections = results.pandas().xyxy[0]  # Pandas DataFrame with detections
#     detected_classes = detections['name']
    
#     # Count vehicles
#     vehicle_count = sum(1 for obj in detected_classes if obj in VEHICLE_CLASSES)
    
#     # Render the results on the image
#     annotated_img = results.render()[0]  # Render detections on the image
    
#     # Fix for OpenCV: Make the rendered image writable
#     annotated_img = annotated_img.copy()
    
#     # Write the vehicle count on the top-left corner
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(annotated_img, f"Vehicles: {vehicle_count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     # Display the output image
#     cv2.imshow("Detected Vehicles", annotated_img)
#     cv2.waitKey(0)  # Wait for a key press to close the window
#     cv2.destroyAllWindows()
    
#     # Save the annotated image
#     output_path = image_path.replace(".jpg", "_output.jpg")  # Save with "_output" appended
#     cv2.imwrite(output_path, annotated_img)
    
#     return vehicle_count, output_path

# def main():
#     # Load YOLOv5 model
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')  # Load YOLOv5 small model weights
    
#     # Input image paths
#     image1 = "image1.jpg"  # Replace with the full path of the first image
#     image2 = "image2.jpg"  # Replace with the full path of the second image
    
#     # Process each image
#     print("Processing Image 1...")
#     count1, output1 = process_image(image1, model)
#     print(f"Image 1: {count1} vehicles detected. Annotated image saved to {output1}")
    
#     print("Processing Image 2...")
#     count2, output2 = process_image(image2, model)
#     print(f"Image 2: {count2} vehicles detected. Annotated image saved to {output2}")

# if __name__ == "__main__":
#     main()






import torch
import cv2


VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

# Green light time per vehicle (in seconds)
TIME_PER_VEHICLE = 2.5

def process_image(image_path, model):
    """
    Processes an image, detects vehicles, displays the output, saves the annotated image, 
    and calculates the green light time based on the number of vehicles detected.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Perform YOLOv5 inference
    results = model(img)
    
    # Extract detections
    detections = results.pandas().xyxy[0]  # Pandas DataFrame with detections
    detected_classes = detections['name']
    
    # Count vehicles
    vehicle_count = sum(1 for obj in detected_classes if obj in VEHICLE_CLASSES)
    
    # Calculate green light time
    green_light_time = vehicle_count * TIME_PER_VEHICLE
    
    # Render the results on the image
    annotated_img = results.render()[0]  # Render detections on the image
    
    # Fix for OpenCV: Make the rendered image writable
    annotated_img = annotated_img.copy()
    
    # Write the vehicle count and green light time on the top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_img, f"Vehicles: {vehicle_count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_img, f"Green Light: {green_light_time} sec", (10, 60), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the output image
    cv2.imshow("Detected Vehicles", annotated_img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    
    # Save the annotated image
    output_path = image_path + "_output.jpg"  # Save with "_output" appended
    cv2.imwrite(output_path, annotated_img)
    
    return vehicle_count, green_light_time, output_path

def main():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')  # Load YOLOv5 small model weights
    
    # Input image paths
    image1 = "image1.jpg" 
    image2 = "image2.jpg"  
    
    # Process each image
    print("Processing Image 1...")
    count1, green_time1, output1 = process_image(image1, model)
    print(f"Image 1: {count1} vehicles detected. Green light time: {green_time1} seconds. Annotated image saved to {output1}")
    
    print("Processing Image 2...")
    count2, green_time2, output2 = process_image(image2, model)
    print(f"Image 2: {count2} vehicles detected. Green light time: {green_time2} seconds. Annotated image saved to {output2}")

if __name__ == "__main__":
    main()
+
