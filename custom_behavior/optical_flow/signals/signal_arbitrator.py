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
        self.fallback_image_conf = self.fallback_image_conf#self.min_image_conf * 0.6 #0.9   # например 0.5 → 0.45

    def select(self, channels: dict[Channel, ChannelConfidence]) -> Channel | tuple[Channel, Channel] | None:
        """
        Логика выбора каналов:
          1. Сначала пытаемся взять оба IMAGE_X + IMAGE_Y (самый точный режим)
          2. Если оба не проходят — пробуем fallback на один из них (если достаточно уверенности)
          3. Если и fallback не прошёл — смотрим ANGLE или OMEGA
          4. Иначе → None (HOLD)
        """
        cx = channels.get(Channel.IMAGE_X)
        cy = channels.get(Channel.IMAGE_Y)
        angle = channels.get(Channel.ANGLE)
        omega = channels.get(Channel.OMEGA)

        decision = None

        # ───────────────────────────────────────────────────────────────
        # 1. Основной приоритет — оба канала позиции одновременно
        # ───────────────────────────────────────────────────────────────
        image_ok_both = (
            cx is not None and
            cy is not None and
            min(cx.value, cy.value) >= self.min_image_conf
        )

        if image_ok_both:
            decision = (Channel.IMAGE_X, Channel.IMAGE_Y)
            print("ARBITRATOR → BOTH")
            return decision

        # ───────────────────────────────────────────────────────────────
        # 2. Fallback — используем только один канал позиции,
        #    если он достаточно уверенный (даже если второй слабый или отсутствует)
        # ───────────────────────────────────────────────────────────────
        fallback_threshold = self.min_image_conf * 0.85   # можно подкрутить: 0.7–0.9

        image_ok_x_only = (
            cx is not None and
            cx.value >= fallback_threshold
        )

        image_ok_y_only = (
            cy is not None and
            cy.value >= fallback_threshold
        )

        if image_ok_x_only:
            decision = Channel.IMAGE_X
            print(f"ARBITRATOR → X only (conf={cx.value:.3f})")
            # telemetry.emit("ARBITRATOR_FALLBACK", channel="IMAGE_X_ONLY", conf=cx.value)
            return decision

        if image_ok_y_only:
            decision = Channel.IMAGE_Y
            print(f"ARBITRATOR → Y only (conf={cy.value:.3f})")
            # telemetry.emit("ARBITRATOR_FALLBACK", channel="IMAGE_Y_ONLY", conf=cy.value)
            return decision

        # ───────────────────────────────────────────────────────────────
        # 3. Вторичные каналы (angle, omega) — без изменений
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
            print(f"ARBITRATOR → angle only (conf={angle.value:.3f})")
            return decision

        if omega_ok:
            decision = Channel.OMEGA
            print(f"ARBITRATOR → omega only (conf={omega.value:.3f})")
            return decision

        # ───────────────────────────────────────────────────────────────
        # Ничего подходящего не нашлось → HOLD
        # ───────────────────────────────────────────────────────────────
        return None