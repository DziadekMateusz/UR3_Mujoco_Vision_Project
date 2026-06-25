# UR3 Vision-Guided Object Recognition and Manipulation System

This project combines computer vision, machine learning and robotic simulation to create an intelligent object recognition and manipulation pipeline for a UR3 robotic arm in MuJoCo.

The vision subsystem uses a two-stage detection architecture. A custom YOLO model performs real-time object detection, while a CNN classifier validates each detection to reduce false positives. The system is trained to recognize three object categories: earphones, computer mouse, and SODIMM RAM module.

After a valid object is detected, the classification result is passed to the robotic control module. The robot operates in a MuJoCo simulation environment containing a UR3 manipulator and predefined target locations corresponding to each object class.

Robot motion is generated using Jacobian-based inverse kinematics, allowing the end-effector to move toward class-specific target positions in Cartesian space.

The project includes modules for dataset preparation, YOLO training and validation, CNN training and evaluation, real-time inference and robotic simulation. Designed and created as a final project for smart materials and methods lessons at university.
