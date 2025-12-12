import numpy as np

class TagGeometry:
    @staticmethod
    def px_size_from_corners(corners):
        d01 = np.linalg.norm(corners[0] - corners[1])
        d12 = np.linalg.norm(corners[1] - corners[2])
        d23 = np.linalg.norm(corners[2] - corners[3])
        d30 = np.linalg.norm(corners[3] - corners[0])
        return 0.25 * (d01 + d12 + d23 + d30)

    @staticmethod
    def pose_height(pose_t):
        try:
            return abs(float(pose_t[2, 0]))
        except Exception:
            return None

    @staticmethod
    def pose_cos_z(pose_R):
        try:
            return abs(float(pose_R[:, 2][2]))
        except Exception:
            return None

    @staticmethod
    def height_from_px(px_size, cos_theta, focal_length_px, tag_size_m):
        if px_size is None or cos_theta is None or cos_theta < 0.01:
            return None
        corrected = px_size * cos_theta
        if corrected < 1:
            return None
        return (focal_length_px * tag_size_m) / corrected
