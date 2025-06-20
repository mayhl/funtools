# NOTE: _ Avoids subsequent import overides
from .core import ElementInterface as _Interface

from ...link.hvbokeh import Native as _Native

import param
import panel as pn


class LineElement(_Interface):

    color = param.Color(default="blue")
    width = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        softbounds=(0, 10),
    )
    alpha = param.Magnitude(
        default=1,
    )

    _LABEL_ = "Line"
    _WIDGET_CONFIGS_ = dict(
        color=dict(
            # Linking kwargs
            widget=pn.widgets.ColorPicker,
            linker=_Native("value", "glyph.line_color"),
            # target_attr = 'glyph.line_color',
            # source_attr = 'value',
            # Widget kwargs
            name="Color",
            # description = "Color of plot line.",
        ),
        width=dict(
            # Linking kwargs
            widget=pn.widgets.FloatSlider,
            linker=_Native("value", "glyph.line_width"),
            # target_attr = 'glyph.line_width',
            # source_attr = 'value',
            # Widget kwargs
            name="Width",
            # description = "Thickness of plot line.",
        ),
        alpha=dict(
            # Linking kwargs
            widget=pn.widgets.FloatSlider,
            linker=_Native("value", "glyph.line_alpha"),
            # target_attr = 'glyph.line_alpha',
            # source_attr = 'value',
            # Widget kwargs
            name="Alpha",
            # description = "Transparency of plot line.",
        ),
    )


# [ 'line_join', 'line_cap', 'line_dash', 'line_dash_offset', 'selection_line_color', 'nonselection_line_color', 'muted_line_color', 'hover_line_color', 'selection_line_alpha', 'nonselection_line_alpha', 'muted_line_alpha', 'hover_line_alpha', 'selection_color', 'nonselection_color', 'muted_color', 'hover_color', 'selection_alpha', 'nonselection_alpha', 'muted_alpha', 'hover_alpha', 'selection_line_width', 'nonselection_line_width', 'muted_line_width', 'hover_line_width', 'selection_line_join', 'nonselection_line_join', 'muted_line_join', 'hover_line_join', 'selection_line_cap', 'nonselection_line_cap', 'muted_line_cap', 'hover_line_cap', 'selection_line_dash', 'nonselection_line_dash', 'muted_line_dash', 'hover_line_dash', 'selection_line_dash_offset', 'nonselection_line_dash_offset', 'muted_line_dash_offset', 'hover_line_dash_offset']


class FillElement(_Interface):

    color = param.Color(default="blue")
    alpha = param.Magnitude(
        default=1,
    )

    _LABEL_ = "Fill"
    _WIDGET_CONFIGS_ = dict(
        color=dict(
            # Linking kwargs
            widget=pn.widgets.ColorPicker,
            linker=_Native("value", "glyph.fill_color"),
            # target_attr = 'glyph.fill_color',
            # source_attr = 'value',
            # Widget kwargs
            name="Color",
            # description = "Color of plot fill.",
        ),
        alpha=dict(
            # Linking kwargs
            widget=pn.widgets.FloatSlider,
            linker=_Native("value", "glyph.fill_alpha"),
            # target_attr = 'glyph.fill_alpha',
            # source_attr = 'value',
            # Widget kwargs
            name="Alpha",
            # description = "Transparency of plot fill.",
        ),
    )


# ['fill_color', 'fill_alpha', 'selection_fill_color', 'nonselection_fill_color', 'muted_fill_color', 'hover_fill_color', 'selection_fill_alpha', 'nonselection_fill_alpha', 'muted_fill_alpha', 'hover_fill_alpha']


class TextElement(_Interface):
    pass


# 'text_font', 'text_font_size', 'text_font_style', 'text_color', 'text_alpha', 'text_align', 'text_baseline'
