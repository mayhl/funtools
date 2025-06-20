import holoviews as hv

from matplotlib.colors import LinearSegmentedColormap
import cmocean
import numpy as np


class ColorBar:

    def __init__(
        self,
        provider: str,
        name: str,
        size: int = 256,
        vmin: float | None = None,
        vmax: float | None = None,
        vmid: float | None = None,
    ) -> None:

        if provider == "cmocean":
            base_name, *_ = name.split("_")
            if not base_name in cmocean.cm.cmapnames:
                raise TypeError(f"Invalid cmocean colormap: {name}")

            self._cm = cmocean.cm.cmap_d[name]
        else:
            cmaps = hv.plotting.util.list_cmaps(
                provider=provider, records=True, reverse=False
            )
            cmaps = [c for c in cmaps if c.name == name]

            if len(cmaps) == 0:
                raise TypeError(f"Invalid {provider} colormap: {name}")

            cm = cmaps[0]
            cm = hv.plotting.util.process_cmap(
                cm.name, provider=cm.provider, ncolors=size
            )

            def convert(c):
                r, g, b = [s / 256 for s in hv.plotting.util.hex2rgb(c)]
                return r, g, b, 1.0

            self._cm = LinearSegmentedColormap.from_list(
                name, colors=[convert(c) for c in cm]
            )

        is_vmin = not vmin is None
        is_vmax = not vmax is None

        if not is_vmin == is_vmax:
            raise TypeError("vmin and vmax must both be specfied")

        self._size = size

        if is_vmin:
            self.scale(vmin, vmax, vmid)
        else:
            self._colors = None
            self._clim = None

    def scale(self, vmin: float, vmax: float, vmid: float | None = None) -> None:

        if vmin > vmax:
            raise ValueError("vmin is larger than vmax")

        if vmid is None:
            vmid = (vmin + vmax) / 2
        else:
            if vmid <= vmin:
                raise ValueError("vmid is smaller than vmin")
            if vmid >= vmax:
                raise ValueError("vmid is larger than vmax")

        xi = [vmin, vmid, vmax]
        si = [0, 0.5, 1.0]

        x = np.linspace(vmin, vmax, self._size)
        s = np.interp(x, xi, si)

        self._colors = [self._cm(s0) for s0 in s]
        self._clim = (vmin, vmax)

    def to_rgba(self) -> list[str]:

        if self._colors is None:
            self._colors = self._cm(np.linspace(0, 1, self._size))

        def convert(r: float, g: float, b: float, a: float) -> str:
            r, g, b = [int(s * 256) for s in [r, g, b]]
            return f"rgba({r:d},{g:d},{b:d},{a:f})"

        return [convert(*c) for c in self._colors]

    def to_holoviews(self) -> dict:

        if self._clim is None:
            return {"cmap": self.to_rgba()}
        else:
            return {"cmap": self.to_rgba(), "clim": self._clim}
