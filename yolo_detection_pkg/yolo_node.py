# Code written by Owen Darden, Converted to ROS node by Jalen Beeman
# URC 2024 Algorithms
# Code is based upon the following link: https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge, CvBridgeError
import math
import numpy as np
import cv2
from ultralytics import YOLO  # Ensure proper import based on your YOLO setup
from ros2_aruco_interfaces.msg import ArucoMarkers

from rclpy.qos import qos_profile_sensor_data



class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        # Declare Parameters
        # Declare parameters individually
        self.declare_parameter('camera_topics', [ '/logitech_18/image_raw']) #'/logitech_01/image_raw', '/logitech_03/image_raw', '/logitech_19/image_raw',
        self.declare_parameter('pose_topic', '/object_poses')
        self.declare_parameter('model_path', '/home/heimdall/workspace-heimdall/src/autonomy/input/yolo_detection_pkg/yolo_models/Datasetv3Medium1280.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('focal_length', 1420.0)
        self.declare_parameter('diagonal_fov', math.radians(78.0))
        self.declare_parameter('init_height', 1080)
        self.declare_parameter('init_width', 1920) # pixels
        self.declare_parameter('ArucoMarker_height', 0.20) # meters
        self.declare_parameter('Hammer_height', 0.36)
        self.declare_parameter('Keyboard_height', 0.16)
        self.declare_parameter('WaterBottle_height', .215)

        # Load parameters
        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.focal_length = self.get_parameter('focal_length').get_parameter_value().double_value
        self.diagonal_fov = self.get_parameter('diagonal_fov').get_parameter_value().double_value
        self.height = self.get_parameter('init_height').get_parameter_value().integer_value
        self.width = self.get_parameter('init_width').get_parameter_value().integer_value
        self.aruco_marker_height = self.get_parameter('ArucoMarker_height').get_parameter_value().double_value
        self.hammer_height = self.get_parameter('Hammer_height').get_parameter_value().double_value
        self.keyboard_height = self.get_parameter('Keyboard_height').get_parameter_value().double_value
        self.water_bottle_height = self.get_parameter('WaterBottle_height').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        

        # Image bridge
        self.bridge = CvBridge()

        # YOLO model
        self.model = YOLO(self.model_path)

        # Subscribers and Publisher
        self.subscribers = [self.create_subscription(Image, topic, self.image_callback, qos_profile_sensor_data) for topic in self.camera_topics]
        self.pose_pub = self.create_publisher(ArucoMarkers, self.pose_topic, qos_profile_sensor_data)
        self.pose2_pub = self.create_publisher(PoseArray, "objects", 10)

        # initalization
        self.classNames = ["ArucoMarker", "Hammer", "Keyboard", "WaterBottle"]
        mult_factor = float(math.lcm(self.height, self.width))
        self.aspect_ratio = [mult_factor/float(self.width), mult_factor/float(self.height)]
        diagonal_size = math.sqrt(self.aspect_ratio[0]**2 + self.aspect_ratio[1]**2)
        self.hori_fov = 2 * math.atan(math.tan(self.diagonal_fov/2)* self.aspect_ratio[0] / (diagonal_size))
        self.vert_fov = 2 * math.atan(math.tan(self.diagonal_fov/2)* self.aspect_ratio[1] / (diagonal_size))
        print(self.hori_fov)
        print(self.vert_fov)


        # Calculate the diagonal resolution
        diagonal_resolution = math.sqrt(float(self.width)**2 + float(self.height)**2)

        # Calculate focal length in pixels
        self.focal_length = diagonal_resolution / (2 * math.tan(self.diagonal_fov / 2))
        

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
            return

        markers = ArucoMarkers()
        markers.header.stamp = msg.header.stamp
        markers.header.frame_id = msg.header.frame_id

        posearray = PoseArray()
        posearray.header.stamp = msg.header.stamp
        posearray.header.frame_id = msg.header.frame_id

        # Run the model on the image and save the results
        results = self.model(img, stream=True,show=False)
        # Otherwise trace through the 
        for r in results:
            boxes = r.boxes
            self.get_logger().info("Result x")
            for box in boxes:
                # Get Confidence Value
                confidence = math.ceil((box.conf[0]*100))/100

                # Skip if bouding box is not within the threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Bounding Box Coordinates in integer
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                self.get_logger().info("hello")
                self.get_logger().info(f"Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")


                # Configure Box Properties
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get Class Name
                cls = int(box.cls[0])

                # ---------------------- Distance Calculations ----------------------------
                # Assuming object is a rectangle where the width is greater than the height

                # Gets the x and y lengths in pixels
                delta_x = x2-x1
                delta_y = y2-y1

                # Object Height assuming bounding box is rectangular
                object_height = delta_y if delta_x < delta_y else delta_x

                # Checks to see if X is the width, use Y in calculations (Ordered for Optimization)
                if (cls == 0): # Aruco Marker
                    est_distance = (self.aruco_marker_height  * self.focal_length) / float(object_height)
                elif (cls == 1): # Hammer
                    est_distance = (self.hammer_height * self.focal_length) / float(object_height)
                elif (cls == 3): # Water Bottle
                    est_distance = (self.water_bottle_height * self.focal_length) / float(object_height)
                elif (cls == 2): # Keyboard
                    est_distance = (self.keyboard_height * self.focal_length) / float(object_height)
                else: # This should never occur
                    self.get_logger().error("Class did not exist")
                    est_distance = -1.0

                print(est_distance)

                # ------------------------- Angle Calculations ----------------------------
                # Horizontal Angle of a pixel (x,y) = ((x - W/2)/(W/2))(HFOV/2)
                # Vertical Angle of a pixel (x,y) = ((y - H/2)/(H/2))(VFOV/2)

                # Calculate the center of the bounding box (x,y)
                x_center = float(x2+x1)/2
                y_center = float(y2+y1)/2

                # Calculate the center of the image
                img_center = [float(self.width)/2, float(self.height)/2]

                # Calculate the horizontal and vertical angle 
                h_angle = ((x_center - img_center[0])/(img_center[0]))*(self.hori_fov/2)
                v_angle = ((y_center - img_center[1])/(img_center[1]))*(self.vert_fov/2)

             

                # -------------------------------------------------------------------------
                # z forward, y down, x across camera coordinate frame
                pose = Pose()
                pose.position.x = est_distance * math.cos(v_angle) * math.sin(h_angle)
                pose.position.y = est_distance * math.sin(v_angle)
                pose.position.z = est_distance * math.cos(v_angle) * math.cos(h_angle)

                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0

                markers.poses.append(pose)
                markers.marker_ids.append(cls)

                posearray.poses.append(pose)

        if markers.poses:
            self.pose_pub.publish(markers)
            self.pose2_pub.publish(posearray)
        

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
