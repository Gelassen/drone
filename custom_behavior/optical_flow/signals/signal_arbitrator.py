from custom_behavior.optical_flow.models.signal_model import Channel, ChannelConfidence
from custom_behavior.optical_flow.models.signal_model import ArbitratorThresholds
import json  # если нужно логировать в json-формате


class Arbitrator:

    def __init__(
        self,
        config: ArbitratorThresholds
    ):
        self.min_image_conf = config.min_image_conf
        self.min_angle_conf = config.min_angle_conf
        self.min_omega_conf = config.min_omega_conf
        
        # порог для fallback-режима — немного ниже основного, чтобы не пропускать совсем слабый сигнал
        self.fallback_image_conf = self.min_image_conf * 0.9   # например 0.5 → 0.45

    def select(self, channels: dict[Channel, ChannelConfidence]) -> Channel | tuple[Channel, Channel] | None:
        """
        Возвращает:
        - (IMAGE_X, IMAGE_Y)       — оба канала прошли gating и достаточно уверены
        - IMAGE_X                  — только X прошёл, Y нет или слабый
        - IMAGE_Y                  — только Y прошёл, X нет или слабый
        - ANGLE                    — angle прошёл
        - OMEGA                    — omega прошёл
        - None                     — HOLD (ничего не выбрано)
        """
        cx = channels.get(Channel.IMAGE_X)
        cy = channels.get(Channel.IMAGE_Y)
        angle = channels.get(Channel.ANGLE)
        omega = channels.get(Channel.OMEGA)

        decision = None

        # ───────────────────────────────────────────────────────────────
        # 1. Основной режим — оба изображения канала достаточно уверены
        # ───────────────────────────────────────────────────────────────
        image_ok_both = (
            cx is not None and
            cy is not None and
            min(cx.value, cy.value) >= self.min_image_conf
        )

        if image_ok_both:
            decision = (Channel.IMAGE_X, Channel.IMAGE_Y)
            # telemetry.emit("ARBITRATOR_DECISION", command="BOTH_IMAGE")  # если логируешь
            return decision

        # ───────────────────────────────────────────────────────────────
        # 2. Fallback — только один канал изображения, но достаточно уверенный
        # ───────────────────────────────────────────────────────────────
        image_ok_x_only = (
            cx is not None and
            cx.value >= self.fallback_image_conf
            # и cy либо отсутствует, либо слишком слабый — не проверяем cy.value
        )

        image_ok_y_only = (
            cy is not None and
            cy.value >= self.fallback_image_conf
        )

        if image_ok_x_only and not image_ok_y_only:
            decision = Channel.IMAGE_X
            # telemetry.emit("ARBITRATOR_FALLBACK", channel="IMAGE_X_ONLY", conf=cx.value)
            return decision

        if image_ok_y_only and not image_ok_x_only:
            decision = Channel.IMAGE_Y
            # telemetry.emit("ARBITRATOR_FALLBACK", channel="IMAGE_Y_ONLY", conf=cy.value)
            return decision

        # ───────────────────────────────────────────────────────────────
        # 3. Angle и Omega — без изменений
        # ───────────────────────────────────────────────────────────────
        angle_ok = (
            angle is not None and
            angle.value >= self.min_angle_conf
        )

        omega_ok = (
            omega is not None and
            omega.value >= self.min_omega_conf
        )

        if angle_ok:
            decision = Channel.ANGLE
            return decision

        if omega_ok:
            decision = Channel.OMEGA
            return decision

        # ───────────────────────────────────────────────────────────────
        # Ничего не выбрано → HOLD
        # ───────────────────────────────────────────────────────────────
        return None