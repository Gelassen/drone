from .tag_geometry import TagGeometry

class TagSelector:
    def select_best(self, tags):
        if not tags:
            return None
        return max(tags, key=lambda t: TagGeometry.px_size_from_corners(t.corners))
