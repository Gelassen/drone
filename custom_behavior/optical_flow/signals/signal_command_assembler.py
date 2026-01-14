
from custom_behavior.optical_flow.models.signal_model import (
    Channel,
    ManagingCommand
)

class CommandAssembler:

    YAW_MAX = 0.8  # rad/s ≈ 45°/s

    def __init__(self):
        print("CommandAssembler::init")

    def signals_to_command(self, scaled_commands):
        # --- Assemble final velocity command ---
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw = 0.0

        if scaled_commands:
            if Channel.IMAGE_X in scaled_commands:
                vx = scaled_commands[Channel.IMAGE_X]

            if Channel.IMAGE_Y in scaled_commands:
                vy = scaled_commands[Channel.IMAGE_Y]

            if Channel.ANGLE in scaled_commands:
                yaw += scaled_commands[Channel.ANGLE]

            if Channel.OMEGA in scaled_commands:
                yaw += scaled_commands[Channel.OMEGA]

            yaw = self.clamp(yaw, -self.YAW_MAX, self.YAW_MAX)

    
        return ManagingCommand(
            velocity_x=vx, 
            velocity_y=vy, 
            velocity_z=vz, 
            yaw=yaw
        )

    def clamp(x, min_val, max_val):
        return max(min_val, min(x, max_val))
