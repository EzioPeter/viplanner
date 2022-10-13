#!/usr/bin/env python3
import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
import copy
import numpy as np
from sensor_msgs.msg import Image
import torchvision.transforms as transforms
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy

rospack = rospkg.RosPack()
pack_path = rospack.get_path('dvf_planner_node')
planner_path = os.path.join(pack_path,'dvf_planner')
sys.path.append(pack_path)
sys.path.append(planner_path)


from dvf_planner.rosutil import ROSArgparse
from dvf_planner import traj_opt, traj_viz

class InterestNode:
    def __init__(self, args, transform):
        super(InterestNode, self).__init__()
        self.config(args)
        self.transform = transform

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_viz = traj_viz.TrajViz(os.path.join(*[planner_path, 'data']), is_sim=args.is_sim, cameraTilt=0.0)
        self.traj_generate = traj_opt.TrajOpt()

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        
        rospy.Subscriber(self.depth_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)

        self.img_pub = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.path_pub  = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", Path, queue_size=10)

        self.tf_listener = tf.TransformListener()
        rospy.loginfo("DVF Planner Ready.")
        

    def config(self, args):
        self.main_freq   = args.main_freq
        self.model_save  = args.model_save
        self.depth_topic = args.depth_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic
        self.image_topic = args.image_topic
        self.frame_id    = args.robot_id
        self.world_id    = args.world_id
        self.uint_type   = args.uint_type
        self.image_flip  = args.image_flip
        self.conv_dist   = args.conv_dist  
        # fear reaction
        self.is_fear_act = args.is_fear_act
        self.buffer_size = args.buffer_size
        self.ang_thred   = args.angular_thred
        self.track_dist  = args.track_dist
        return 

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning:
                # main planning starts
                with torch.no_grad():
                    self.preds, self.fear = self.net(self.img, self.goal)
                    self.waypoints = self.traj_generate.TrajGeneratorFromPFreeRot(self.preds, step=0.1)
                # check goal less than converage range
                goal_np = self.goal[0, :].cpu().detach().numpy()
                if (np.sqrt(goal_np[0]**2 + goal_np[1]**2) < self.conv_dist):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    rospy.loginfo("Goal Arrived")
                if self.is_fear_act:
                    is_track_ahead = self.isForwardTraking(self.waypoints)
                    self.fearPathDetection(self.fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn("current path prediction is invaild.")
                self.pubPath(self.waypoints, self.is_goal_init)
                # visualize image
                self.pubRenderImage(self.preds, self.waypoints, self.odom, self.goal, self.fear, self.img)
            r.sleep()
        rospy.spin()

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        fear_path = Path()
        if is_goal_init:
            for p in waypoints.squeeze(0):
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path.poses.append(pose)
        # add header
        path.header.frame_id = fear_path.header.frame_id = self.frame_id
        path.header.stamp = fear_path.header.stamp = self.image_time
        # publish fear path
        if self.is_fear_reaction:
            fear_path.poses = copy.deepcopy(path.poses)
            path.poses = path.poses[:1]
        # publish path
        self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)
        return

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return None

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if np.all(phead != None) and phead.dot(xhead) > 1.0 - self.ang_thred:
            return True
        return False

    def goalCallback(self, msg):
        rospy.loginfo("Recevied a new goal")
        self.goal_pose = msg
        self.is_goal_init = True
        # reset fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        return

    def pubRenderImage(self, preds, waypoints, odom, goal, fear, image):
        image = self.traj_viz.VizImages(preds, waypoints, odom, goal, fear, image)[0]
        ros_img = ros_numpy.msgify(Image, image, encoding='rgb8')
        self.img_pub.publish(ros_img)
        return None

    def imageCallback(self, msg):
        # rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        self.image_time = msg.header.stamp
        frame = ros_numpy.numpify(msg)
        frame[~np.isfinite(frame)] = 0
        if self.uint_type:
            frame = frame / 1000.0
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
        # img.show()
        img = PIL.Image.fromarray(frame)
        if self.image_flip:
            img = img.transpose(PIL.Image.ROTATE_180)
        img = self.transform(img)[None, ...]
        if self.is_goal_init:
            p_in_vehicle = self.goal_pose
            p_in_vehicle.header.stamp = rospy.Time(0)
            try:
                p_in_vehicle = self.tf_listener.transformPoint(self.frame_id, p_in_vehicle)
                (odom, ori) = self.tf_listener.lookupTransform(self.world_id, self.frame_id, rospy.Time(0))
                odom.extend(ori)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logerr("Fail to transfer the goal into vehicle frame.")
                return
            p_in_vehicle = torch.tensor([p_in_vehicle.point.x, p_in_vehicle.point.y, p_in_vehicle.point.z], dtype=torch.float32)[None, ...]
        else:
            return
        odom = torch.tensor(odom, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            self.odom  = odom.cuda()
            self.img = img.cuda()
            self.goal  = p_in_vehicle.cuda()
        else:
            self.odom  = odom, self.im = img, self.goal = p_in_vehicle
        self.ready_for_planning = True
        return

if __name__ == '__main__':

    node_name = "dvf_planner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('main_freq',     type=int,   default=5,                          help="frequency of path planner")
    parser.add_argument('model_save',    type=str,   default='/models/plannernet.pt',    help="read model")
    parser.add_argument('crop_size',     type=tuple, default=[360,640],                  help='image crop size')
    parser.add_argument('uint_type',     type=bool,  default=False,                      help="image in uint type or not")
    parser.add_argument('depth_topic',   type=str,   default='/rgbd_camera/depth/image', help='depth image ros topic')
    parser.add_argument('goal_topic',    type=str,   default='/way_point',               help='goal waypoint ros topic')
    parser.add_argument('path_topic',    type=str,   default='/path',                    help='DVF path topic')
    parser.add_argument('image_topic',   type=str,   default='/path_image',              help='DVF image view topic')
    parser.add_argument('robot_id',      type=str,   default='base',                     help='robot TF frame id')
    parser.add_argument('world_id',      type=str,   default='odom',                     help='world TF frame id')
    parser.add_argument('is_sim',        type=bool,  default=False,                      help='is in simulation setting')
    parser.add_argument('image_flip',    type=bool,  default=True,                       help='is the image fliped')
    parser.add_argument('conv_dist',     type=float, default=0.5,                        help='converge range to the goal')
    parser.add_argument('is_fear_act',   type=bool,  default=True,                       help='is open fear action or not')
    parser.add_argument('buffer_size',   type=int,   default=10,                         help='buffer size for fear reaction')
    parser.add_argument('angular_thred', type=float, default=1.0,                        help='angular thred for turning')
    parser.add_argument('track_dist',    type=float, default=0.5,                        help='look ahead distance for path tracking')

    args = parser.parse_args()
    args.model_save = planner_path + args.model_save

    depth_transform = transforms.Compose([
        transforms.Resize(tuple(args.crop_size)),
        transforms.ToTensor()])

    node = InterestNode(args, depth_transform)

    node.spin()
