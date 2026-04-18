import numpy as np
from scipy import spatial


class KDTree(spatial.KDTree):

    def __init__(
        self,
        data,
        leafsize=10,
        compact_nodes=True,
        copy_data=False,
        balanced_tree=True,
        boxsize=None,
        grid_bounds=None,
    ):
        super().__init__(
            data[:, :2], leafsize, compact_nodes, copy_data, balanced_tree, boxsize
        )

        self._raw_data = data

        self._kwargs = {
            "leafsize": leafsize,
            "compact_nodes": compact_nodes,
            "copy_data": copy_data,
            "balanced_tree": balanced_tree,
            "boxsize": boxsize,
        }

        x0 = data[:, 0].min()
        x1 = data[:, 0].max()
        y0 = data[:, 1].min()
        y1 = data[:, 1].max()

        if not grid_bounds is None:
            self._grid_bounds = grid_bounds
            pass
            u0, v0, u1, v1 = grid_bounds

            if x1 - x0 > 0:

                if u0 < x1:
                    x0 = max([x0, u0])

                if u1 > x0:
                    x1 = min([x1, u1])

            if y1 - y0 > 0:

                if v0 < y1:
                    y0 = max([y0, v0])

                if v1 > y0:
                    y1 = min([y1, v1])

        else:
            self._grid_bounds = x0, y0, x1, y1

    def _construct_equispaced(self, bounds, n_target: int):

        # assert n_target > 0
        x0, y0, x1, y1 = bounds

        m_target = n_target**2

        lx = x1 - x0
        ly = y1 - y0

        # assert lx > 0
        # assert ly > 0
        # assert m_target > 0
        ds = np.sqrt(lx * ly / m_target)

        nx = int(np.round(lx / ds))
        ny = int(np.round(ly / ds))

        if nx == 0:
            nx = 1
        if ny == 0:
            ny = 1

        dx = lx / nx
        dy = ly / ny

        dr = np.sqrt((dx / 2) ** 2 + (dy / 2) ** 2)
        return dr, dx, dy, nx, ny

    def __reduce__(self):

        args = (
            self.data,
            *self._kwargs.values(),
            self.__dict__["_grid_bounds"],
        )

        return (self.__class__, args)

    def _get_cell_centered_nearest(self, bounds, n_target: int, workers: int = 1):

        dr, dx, dy, nx, ny = self._construct_equispaced(bounds, n_target)

        x0, y0, *_ = bounds
        x = np.arange(nx) * dx + dx / 2 + x0
        y = np.arange(ny) * dy + dy / 2 + y0

        pts = np.vstack([s.flatten() for s in np.meshgrid(x, y)]).T
        _, indices = self.query(pts, distance_upper_bound=dr, workers=workers, p=np.inf)

        filt = indices < self.n

        indices = indices[filt]
        pts = pts[filt]

        filt = filt.reshape((ny, nx))

        return indices, filt, pts, (dx, dy), dr

    def _filter_cell_centered(self, target_count: int, workers: int = 1):
        """Returns subcollection of point indices by selecting the nearest point to each cell-center
        within a recursively refined equipartition grid, ensuring a target point count is
        met within error margins."""
        max_iterations = 20

        # Thresholds for selecting different scaling factors
        # for recusive search. Last threshold for solution.
        thresholds = [0.2, 0.5, 0.75, 0.95]
        # Factors for scaling target linear 1D point count
        factors = [5, 1.5, 1.1, 1.05]

        def get_factor(r):
            for t, f in zip(thresholds, factors):
                if r < t:
                    return f
            return None

        # Target linear 1D point count
        old_target = n_target = 1
        bounds = self._grid_bounds
        for i in range(max_iterations):

            indices, filt, *args = self._get_cell_centered_nearest(
                bounds, n_target, workers=workers
            )
            m = np.sum(filt)
            if m < target_count:
                r = m / target_count
                factor = get_factor(m / target_count)
                if factor is None:
                    break
                n_target *= factor
            else:
                r = target_count / m
                factor = get_factor(target_count / m)
                if factor is None or n_target <= 1:
                    break

                n_target /= factor

            n_target = int(n_target)

            # print(f"Build {i:d} | {r:f} | {n_target:d}")
            if abs(old_target - n_target) == 0:
                break

            old_target = n_target
        return indices, filt, bounds, *args

    def filter_cell_centered(self, target_count: int, workers: int = 1):
        """Returns subcollection of point indices by selecting the nearest point to each cell-center
        within a recursively refined equipartition grid, ensuring a target point count is
        met within error margins."""

        indices, *_ = self._filter_cell_centered(target_count, workers)
        return self._raw_data[indices, :]

    def density_heatmap(self, target_nbins, workers: int = 1):

        args = self._filter_cell_centered(target_nbins, workers=workers)
        _, filt, bounds, pts, _, radius = args

        sizes = np.zeros(filt.shape)

        indices = self.query_ball_point(pts, radius, workers=workers, p=np.inf)
        sizes[filt] = [len(i) for i in indices]

        return np.ma.masked_array(sizes, mask=sizes == 0), bounds, radius

    def filter_cell_indices(self, target_min_size, workers: int = 1):

        max_interations = 20

        def query(nbins):
            args = self._filter_cell_centered(nbins, workers=workers)
            *_, pts, ds, radius = args
            #
            indices = self.query_ball_point(pts, radius, workers=workers, p=np.inf)

            # indices = self.query(
            #    pts, k=3 * target_min_size, distance_upper_bound=radius
            # )
            return pts, indices, ds, min([len(i) for i in indices])

        ratio = 4
        n0 = 1
        n1 = ratio**4

        x0, y0, x1, y1 = self._grid_bounds
        args = self.data, np.arange(self.n), (x1 - x0, y1 - y0)

        for i in range(max_interations):

            # print(f"Init {i:d}")
            *new_args, min_size = query(n1)
            if min_size < target_min_size:
                break

            n0 = n1
            n1 *= ratio
            args = new_args

        for i in range(max_interations):

            # print(f"Refine {i:d}")
            nc = int((n0 + n1) / 2)
            *new_args, min_size = query(nc)
            if min_size < target_min_size:
                n1 = nc
            else:
                n0 = nc
                args = new_args

            if n1 - n0 == 1:
                break

        def create(pt, i):
            bounds = (
                pt[0] - ds[0] / 2,
                pt[1] - ds[1] / 2,
                pt[0] + ds[0] / 2,
                pt[1] + ds[1] / 2,
            )

            if not isinstance(i, list):
                i = [i]

            return KDTree(self._raw_data[i, :], **self._kwargs, grid_bounds=bounds)

        pts, indices, ds = args
        return [create(*a) for a in zip(pts, indices)]

    def get_equipartition_dimensions_by_point_count(
        self, points_per_subgrid, max_interations=10, error_tol=0.1
    ):

        n, _ = self.data.shape
        n0 = n / points_per_subgrid

        def get_median_subpoints(n):
            heatmap, *_ = self.density_heatmap(n)
            return np.median(heatmap[~heatmap.mask].data)

        n_pts1 = n_pts0 = get_median_subpoints(n0)

        for i in range(max_interations):

            n1 = np.sqrt(n_pts0 / points_per_subgrid) * n0
            n_pts1 = get_median_subpoints(n1)

            err = abs(n_pts1 - points_per_subgrid) / points_per_subgrid
            if err < error_tol:
                break

            n0 = n1
            n_pts0 = n_pts1

        if i == max_interations - 1:
            raise ValueError("Max Interations reached.")

        _, dx, dy, nx, ny = self._construct_equispaced(self._grid_bounds, n_pts1)

        x0, y0, *_ = self._grid_bounds
        return dx, dy, nx, ny, x0, y0
