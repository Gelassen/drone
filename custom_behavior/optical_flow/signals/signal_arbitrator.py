from custom_behavior.optical_flow.models.signal_model import Channel, ChannelConfidence
from custom_behavior.optical_flow.models.signal_model import ArbitratorThresholds

class Arbitrator:

    def __init__(
        self,
        config: ArbitratorThresholds
    ):
        self.min_image_conf = config.min_image_conf
        self.min_angle_conf = config.min_angle_conf
        self.min_omega_conf = config.min_omega_conf

    def select(self, channels: dict[Channel, ChannelConfidence]) -> Channel:
        cx = channels.get(Channel.IMAGE_X)
        cy = channels.get(Channel.IMAGE_Y)
        angle = channels.get(Channel.ANGLE)
        omega = channels.get(Channel.OMEGA)

        decision = None

        image_ok = (
            cx is not None and
            cy is not None and
            min(cx.value, cy.value) >= self.min_image_conf
        )

        angle_ok = (
            angle is not None and
            angle.value >= self.min_angle_conf
        )

        omega_ok = (
            omega is not None and
            omega.value >= self.min_omega_conf
        )

        if image_ok:
            decision = (Channel.IMAGE_X, Channel.IMAGE_Y)
        elif angle_ok:
            decision = Channel.ANGLE
        elif omega_ok:
            decision = Channel.OMEGA

        return decision  # None == HOLD
