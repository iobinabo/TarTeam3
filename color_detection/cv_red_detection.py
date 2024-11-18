import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import cv2
import gi
import numpy as np
import sys
import pygame

gi.require_version('Gst', '1.0')
from gi.repository import Gst

drone = System()
SPEED = 0.5 # m/s
DESIRED_BUFFER = 2.5 # m
ZERO_VAL = 0.0
m_velocity = [ZERO_VAL, ZERO_VAL, ZERO_VAL, ZERO_VAL] # [forward, down, right, rotate]

class Color:
    m_low_hue = 0
    m_high_hue = 0
    
    m_low_saturation = 0
    m_high_saturation = 0

    m_low_value = 0
    m_high_value = 0

    m_color_name = ""

    def __init__(self, hueL: int, saturationL: int, valueL: int, hueH: int, saturationH: int, valueH: int, name: str) -> None:
        self.m_low_hue = hueL
        self.m_low_saturation = saturationL
        self.m_low_value = valueL
        
        self.m_high_hue = hueH
        self.m_high_saturation = saturationH
        self.m_high_value = valueH

        self.m_color_name = name
    
    def getLowRange(self):
        return np.array([self.m_low_hue, self.m_low_saturation, self.m_low_value], np.uint8)

    def getHighRange(self):
        return np.array([self.m_high_hue, self.m_high_saturation, self.m_high_value], np.uint8)
    
    def getName(self):
        return self.m_color_name

class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK

def init_pygame():
    print("Initializing pygame...")
    pygame.init()
    pygame.display.set_mode((400, 400))
    print("Pygame initialized.")

def get_key(keyName):
    ans = False
    for event in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

def process_frame(frame):
    # Convert the frame to HSV (Hue, Saturation, Value)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color ranges
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 150, 150])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Combine masks
    red_mask = mask1 + mask2
    mid_x, mid_y = None, None
    biggestArea, contour = 0, None

    # Find contours and draw bounding boxes
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > biggestArea:
            biggestArea = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            mid_x, mid_y, contour = float(x+w)/2, float(y+h)/2, c
        
    if contour is None:
        return None
    return [mid_x, mid_y]

def feet_to_meters(dist:float):
    return dist/3.281 # m

def dist_to_time(dist:float):
    return dist / SPEED # s

def kill_switch():
    velocity = [ZERO_VAL, ZERO_VAL, ZERO_VAL, ZERO_VAL]

async def wait_time(time:float):
    await asyncio.sleep(time)
    kill_switch()

def move_forward(distance:float):
    m_velocity[0] = SPEED
    wait_time(dist_to_time(dist=distance))

def move_up(distance):
    m_velocity[1] = SPEED
    wait_time(dist_to_time(dist=distance))

def lift_up():
    m_velocity[1] = SPEED

async def main():
    print("Connecting to drone...")
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    # Wait for the drone to reach a stable altitude
    await asyncio.sleep(5)

    # Initial setpoint before starting offboard mode
    initial_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    await drone.offboard.set_velocity_body(initial_velocity)

    print("-- Setting offboard mode")
    await drone.offboard.start()

    # Initialize GStreamer video object for capturing the drone's camera feed
    video = Video()

    detected_ids = []  # List to keep track of detected ArUco marker IDs

    while True:
        # Ensure m_velocity[1] is set to ZERO_VAL to hover at the current altitude
        
        # Get keyboard inputs and control the drone
        velocity = m_velocity
        v_curr = VelocityBodyYawspeed(velocity[0], velocity[1], velocity[2], velocity[3])
        await drone.offboard.set_velocity_body(v_curr)

        # If frame is available, display the video feed
        if video.frame_available():
            frame = video.frame()
            frame = np.array(frame)
            bbox = process_frame(frame)
            if bbox is not None:
                print(bbox)
                velocity = [0.0, 0.0, 0.1, 0.0]  # Adjust Z velocity for upward movement
            else:
                print("-- No Object Detected")

                # Stop drone movement if no object detected
                velocity = [0.0, 0.0, 0.0, 0.0]
                
            v_curr = VelocityBodyYawspeed(velocity[0], velocity[1], velocity[2], velocity[3])
            await drone.offboard.set_velocity_body(v_curr)

            # Detect ArUco markers
            #arucoFound = findArucoMarkers(frame)
            cv2.imshow("Drone Camera Stream", frame)
            print(v_curr)

            # # Loop through detected ArUco markers and track IDs
            # if len(arucoFound[0]) != 0:  # Check if any markers are detected
            #     for bbox, id in zip(arucoFound[0], arucoFound[1]):
            #         id_value = int(id[0])  # Convert ID to integer
            #         if id_value not in detected_ids:  # Check if ID is new
            #             detected_ids.append(id_value)  # Add new ID to list
            #             print(f"New ID detected: {id_value}")

            # print(f"Current detected IDs: {detected_ids}")  # Print current detected IDs

        # Check for 'l' key to land the drone
        if get_key("l"):
            print("-- Landing")
            await drone.action.land()
            break

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.1)

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    init_pygame()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
