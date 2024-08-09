# Code written by Owen Darden, Converted to ROS node by Jalen Beeman
# URC 2024 Algorithms
# Code is based upon the following link: https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

from time import sleep
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
from matplotlib import pyplot as plt

class YoloNode(Node):
    def __init__(self):

        super().__init__('yolo_node')
        # Declare Parameters
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('transform_topic', '/object_transform')
        # self.declare_parameter('transform_parent', 'logitech_0')
        self.declare_parameter('calculation_freq', 60.0)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('model', 'Datasetv3Small1280.pt')
        self.declare_parameter('ui_feedback', True)
        self.declare_parameter('focal_length', 1475)
        self.declare_parameter('diagonal_fov', (78.0/180.0) * 3.14)
        self.declare_parameter('init_height', 1080)
        self.declare_parameter('init_width', 1920)
        self.declare_parameter('keyboard_height', 16)
        self.declare_parameter('hammer_height', 12)
        self.declare_parameter('aruco_height', 20)
        self.declare_parameter('water_bottle_height', 21.5)

        # Image holder and Bridge
        self.current_frame = Image()
        self.bridge = CvBridge()

        # Model
        self.model = YOLO(self.get_parameter('model').get_parameter_value().string_value)
        self.classNames = ["ArucoMarker", "Hammer", "Keyboard", "WaterBottle"]

        # Set up image subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').get_parameter_value().string_value,
            self.image_sub_callback,
            10
        )

        # Set up transform publisher
        self.transform_pub = self.create_publisher(
            TransformStamped,
            self.get_parameter('transform_topic').get_parameter_value().string_value,
            10
        )

        # Set up timer
        self.timer = self.create_timer(
            1.0/self.get_parameter('calculation_freq').get_parameter_value().double_value,
            self.timer_callback,
        )

        # Set up all the default variables
        self.calculate()

        # Heights of the Objects # in cm (or we can change it to any unit as long as we keep it consistant)
        self.keyboard_height = self.get_parameter('keyboard_height').get_parameter_value().double_value 
        self.hammer_height = self.get_parameter('hammer_height').get_parameter_value().double_value
        self.aruco_height = self.get_parameter('aruco_height').get_parameter_value().double_value
        self.water_bottle_height = self.get_parameter('water_bottle_height').get_parameter_value().double_value 
        
    # Image Subscriber Callback
    def image_sub_callback(self, msg):

        # Save the incoming frame
        self.current_frame = msg

    # Timer Callback
    def timer_callback(self):

        self.get_logger().info("Timer Callback triggered")

        # Convert the current frame to a usable format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.current_frame, desired_encoding='passthrough')
            bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_YUV2BGR_YUY2)

        except CvBridgeError as e:
            print(e)
            return
        
        # If the conversion was successful, run Yolo
        self.yolo_analyze(bgr_image)

    # Analyze a given cv image
    def yolo_analyze(self, img):

        # Run the model on the image and save the results
        results = self.model(img, stream=True)

        # Otherwise trace through the 
        for r in results:
            boxes = r.boxes
            self.get_logger().info("Result x")
            for box in boxes:
                # Get Confidence Value
                confidence = math.ceil((box.conf[0]*100))/100

                # Skip if bouding box is not within the threshold
                if confidence < self.get_parameter('confidence_threshold').get_parameter_value().double_value:
                    continue
                
                # Bounding Box Coordinates in integer
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

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
                object_height = delta_y if delta_x>delta_y else delta_x

                # Checks to see if X is the width, use Y in calculations (Ordered for Optimization)
                if (cls == 0): # Aruco Marker
                    est_distance = (self.aruco_height * self.focal_length) / (object_height)
                elif (cls == 1): # Hammer
                    est_distance = (self.hammer_height * self.focal_length) / (object_height)
                elif (cls == 3): # Water Bottle
                    est_distance = (self.water_bottle_height * self.focal_length) / (object_height)
                elif (cls == 2): # Keyboard
                    est_distance = (self.keyboard_height * self.focal_length) / (object_height)
                else: # This should never occur
                    self.get_logger().error("Class did not exist")
                    est_distance = -1

                # ------------------------- Angle Calculations ----------------------------
                # Horizontal Angle of a pixel (x,y) = ((x - W/2)/(W/2))(HFOV/2)
                # Vertical Angle of a pixel (x,y) = ((y - H/2)/(H/2))(VFOV/2)

                # Calculate the center of the bounding box (x,y)
                x_center = (x2+x1)/2
                y_center = (y2+y1)/2

                # Calculate the center of the image
                img_center = [self.width/2, self.height/2]

                # Calculate the horizontal and vertical angle 
                h_angle = ((x_center - img_center[0])/(img_center[0]))*(self.hori_fov/2)
                v_angle = ((y_center - img_center[1])/(img_center[1]))*(self.vert_fov/2)

                # -------------------------------------------------------------------------

                object_transform = TransformStamped()
                object_transform.child_frame_id = self.classNames[cls]
                
                # Convert between spherical and cartesian
                if est_distance == 0: est_distance = 1
                object_transform.x = est_distance * math.sin(v_angle) * math.cos(h_angle)
                object_transform.y = est_distance * math.sin(v_angle) * math.sin(h_angle)
                object_transform.z = est_distance * math.cos(v_angle)

                self.transform_pub.publish(object_transform)

    def calculate(self):

        # Get parameter values
        self.height = self.get_parameter('init_height').get_parameter_value().double_value
        self.width = self.get_parameter('init_width').get_parameter_value().double_value
        self.diag_fov = self.get_parameter('diagonal_fov').get_parameter_value().double_value
        
        # Determine the aspect ratio
        mult_factor = math.lcm(self.height, self.width)
        self.aspect_ratio = [int(mult_factor/self.width), int(mult_factor/self.height)]

        # Determine the size of the diagonal
        diagonal_size = math.sqrt(self.aspect_ratio**2 + self.aspect_ratio[1]**2)

        # Horizontal FOV
        self.hori_fov = 2 * math.atan(math.tan(self.diag_fov/2)* self.aspect_ratio[0] / (diagonal_size)) / 3.14 * 180
        # Vertical FOV
        self.vert_fov = 2 * math.atan(math.tan(self.diag_fov/2)* self.aspect_ratio[1] / (diagonal_size)) / 3.14 * 180

def main():
    rclpy.init()
    yolo_node = YoloNode()

    rclpy.spin(yolo_node)

    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()