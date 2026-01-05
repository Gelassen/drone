from models.signal_model import Channel, ChannelConfidence

class Arbitrator:

    def __init__(
        self,
        min_image_conf=0.6,
        min_angle_conf=0.55,
        min_omega_conf=0.5
    ):
        self.min_image_conf = min_image_conf
        self.min_angle_conf = min_angle_conf
        self.min_omega_conf = min_omega_conf

    def select(self, channels: dict[Channel, ChannelConfidence]):
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
