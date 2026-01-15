#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from control_input_msgs.msg import ContactInfo


class ContactInfoMonitor(Node):
    def __init__(self):
        super().__init__('contact_info_monitor')
        
        self.subscription = self.create_subscription(
            ContactInfo,
            '/contact_info',
            self.contact_callback,
            10)
        
        self.get_logger().info('Contact Info Monitor started')
    
    def contact_callback(self, msg):
        self.get_logger().info('=== Contact Info ===')
        self.get_logger().info(f'Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        
        # Print contact flags
        self.get_logger().info(f'Feet Contact:  {list(msg.feet_contact)}')
        self.get_logger().info(f'Gait Contact:  {list(msg.gait_contact)}')
        self.get_logger().info(f'Est Contact:   {list(msg.est_contact)}')
        
        # Print force magnitudes
        self.get_logger().info(f'Force Magnitude: {[f"{f:.2f}" for f in msg.est_force_magnitude]}')
        self.get_logger().info(f'Force Z:         {[f"{f:.2f}" for f in msg.est_force_z]}')
        
        # Print first foot's detailed forces
        self.get_logger().info(f'FR Foot Forces (x,y,z): '
                             f'{msg.est_contact_forces[0]:.2f}, '
                             f'{msg.est_contact_forces[1]:.2f}, '
                             f'{msg.est_contact_forces[2]:.2f}')
        self.get_logger().info('=' * 50)


def main(args=None):
    rclpy.init(args=args)
    monitor = ContactInfoMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
