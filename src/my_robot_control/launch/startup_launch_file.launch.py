import launch
from launch import LaunchDescription
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
   return LaunchDescription([
       LogInfo(
           condition=launch.conditions.IfCondition(LaunchConfiguration('enable_log_info')),
           message="This is a test log message."
       )
   ])

