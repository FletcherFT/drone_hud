#!/usr/bin/env python


import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Imu, Range, CompressedImage
from geometry_msgs.msg import Quaternion
import numpy as np
from scipy.spatial.transform import Rotation
from drone_hud.cfg import HudTextConfig, HudCompassConfig
from dynamic_reconfigure.server import Server
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


class Text:
    def __init__(self, namespace, width, height, channels):
        self._enabled = False
        self._content = None
        self._position = None
        self._scale = None
        self._color = None
        self._thickness = None
        self._alpha = None
        self._layer = np.zeros((height, width, channels + 1), dtype=np.uint8)
        self._server = Server(HudTextConfig, self.update, namespace=namespace)

    def update(self, config, level):
        self._content = config["text_content"]
        self._position = np.array([[config["position_x"]], [config["position_y"]]], dtype=float)
        self._scale = config["scale"]
        self._color = (config["blue"],config["green"],config["red"])
        self._thickness = config["thickness"]
        self._alpha = config["alpha"]
        self._enabled = config["enable"]
        self._update_layer()
        return config

    def _update_layer(self):
        if not self._enabled:
            self._layer = np.zeros_like(self._layer)
            return
        height, width, channels = self._layer.shape
        pos = (self._position * [[width], [height]]).squeeze().astype(int)
        layer = np.zeros((height, width, channels-1), dtype=np.uint8)
        layer = cv2.putText(layer, self._content, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, self._scale, self._color,
                            self._thickness, cv2.LINE_AA, bottomLeftOrigin=False)
        mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY) > 0
        alpha = (np.where(mask, self._alpha, 0.0) * 255.0).astype(np.uint8)
        self._layer = np.concatenate((layer, alpha[..., None]), axis=-1)

    def __call__(self):
        if not self._enabled:
            return np.zeros_like(self._layer)
        return self._layer


class Compass:
    def __init__(self, namespace, width, height, channels):
        self._enabled = False
        self._position = None
        self._scale = None
        self._color = None
        self._thickness = None
        self._shift = None
        self._tip_length = None
        self._alpha = None
        self._R = Rotation.from_quat([0, 0, 0, 1])
        self._layer = np.zeros((height, width, channels + 1), dtype=np.uint8)
        self._imu_topic = rospy.get_param(namespace+"/topic")
        self._imu_timeout = self._imu_timeout = rospy.Duration.from_sec(rospy.get_param(namespace+"/timeout", 1.0))
        self._last = rospy.Time.from_sec(0)
        self._server = Server(HudCompassConfig, self.update, namespace=namespace)

        rospy.Subscriber(self._imu_topic, Imu, self._handle_imu)

    def _handle_imu(self, msg):
        q = msg.orientation
        self._last = msg.header.stamp
        self._R = Rotation.from_quat([q.x, q.y, q.z, q.w])

    def update(self, config, level):
        self._position = np.array([[config["position_x"]], [config["position_y"]]], dtype=float)
        self._scale = config["scale"]
        self._color = (config["blue"],config["green"],config["red"])
        self._thickness = config["thickness"]
        self._shift = config["shift"]
        self._tip_length = config["tip_length"]
        self._alpha = config["alpha"]
        self._enabled = config["enable"]
        self._update_layer()
        return config

    def _update_layer(self):
        if not self._enabled:
            self._layer = np.zeros_like(self._layer)
            return
        height, width, channels = self._layer.shape
        pt1 = (self._position * [[width], [height]])
        yaw = -1 * self._R.as_euler("xyz")[-1]
        pt2 = self._scale * np.array([[np.cos(yaw)], [np.sin(yaw)]]) + pt1
        pt1 = pt1.squeeze().astype(int)
        pt2 = pt2.squeeze().astype(int)
        layer = np.zeros((height, width, channels-1), dtype=np.uint8)
        if (rospy.Time.now() - self._last) < self._imu_timeout:
            layer = cv2.arrowedLine(layer, tuple(pt1), tuple(pt2), self._color,
                                    self._thickness, cv2.LINE_AA, self._shift, self._tip_length)
        else:
            rospy.logwarn_throttle(10.0, "{} | Imu timeout.".format(rospy.get_name()))
            layer = cv2.putText(layer, "?", tuple(pt1), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255),
                                3, cv2.LINE_AA, bottomLeftOrigin=False)
        mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY) > 0
        alpha = (np.where(mask, self._alpha, 0.0) * 255.0).astype(np.uint8)
        self._layer = np.concatenate((layer, alpha[..., None]), axis=-1)

    def __call__(self):
        if not self._enabled:
            self._layer = np.zeros_like(self._layer)
            return
        return self._layer


def parallelize(pool, functions):
    futures = [pool.apply_async(f) for f in functions]
    results = [fut.get() for fut in futures]
    return results


class DroneHud:
    def __init__(self):
        self._width = rospy.get_param("~width")
        self._height = rospy.get_param("~height")
        self._channels = rospy.get_param("~channels", 3)
        self._find_hud_objects()
        n_workers = rospy.get_param("~workers", None)
        n_workers = min(len(self._objects), cpu_count()) if n_workers is None else n_workers
        self._pool = ThreadPool(n_workers)
        image_topic = "image_in"
        self._cvbridge = CvBridge()
        self._annotated_img_pub = rospy.Publisher("image_out/compressed", CompressedImage, queue_size=10)
        rospy.Subscriber(image_topic+"/compressed", CompressedImage, self._handle_compressed_image)

    def _find_hud_objects(self):
        name = rospy.get_name()
        params = [p for p in rospy.get_param_names() if p.startswith(name)]
        objects = set([p.split(rospy.get_name())[1].split("/")[1] for p in params])
        self._objects = []
        notfounds = []
        for o in objects:
            if o.startswith("text"):
                self._objects.append(Text(rospy.get_name()+"/"+o, self._width, self._height, self._channels))
            elif o.startswith("compass"):
                self._objects.append(Compass(rospy.get_name() + "/" + o, self._width, self._height, self._channels))
            else:
                notfounds.append(rospy.get_name()+"/"+o)
        if len(notfounds) > 0:
            rospy.logerr("{} | Params listed could not be matched with objects: {}".format(rospy.get_name(), repr(notfounds)))

    def _handle_compressed_image(self, msg):
        img = self._cvbridge.compressed_imgmsg_to_cv2(msg)
        results = parallelize(self._pool, self._objects)
        img = img.astype(float) / 255.0
        for res in results:
            res = res.astype(float) / 255.0
            img = (1-res[...,-1][...,None])*img + res[...,-1][...,None]*res[:,:,:3]
        img = (img * 255.0).astype(int)
        msg = self._cvbridge.cv2_to_compressed_imgmsg(img)
        self._annotated_img_pub.publish(msg)


def main():
    rospy.init_node("drone_hud_node")
    try:
        obj = DroneHud()
        if rospy.get_param("~test", False):
            imu_pub = rospy.Publisher("imu", Imu, queue_size=10)
            imu_msg = Imu()
            eulers = np.array([[0], [0], [np.pi / 2]])
            eulers = np.where(eulers > 2*np.pi, 0.0, eulers)
            i = 0
            while not rospy.is_shutdown():
                eulers = eulers + [[0], [0], [0.005]]
                imu_msg.header.stamp = rospy.Time.now()
                imu_msg.header.seq = i
                q = Rotation.from_euler("xyz", eulers.squeeze()).as_quat()
                imu_msg.orientation = Quaternion(*q)
                imu_pub.publish(imu_msg)
                i += 1
        else:
            rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
