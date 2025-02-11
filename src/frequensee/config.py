from __future__ import annotations
from collections.abc import Iterable
import re

class Config():

    def __init__(self, options: dict[str, str|int|float|bool|None] = {}):
        '''
        Configuration Class containing data required for the `AudioVisualizer` class.
        An optional dictionary can be passed as initialization argument to modify the properties en masse.
        '''

        self.rgba_regex: re.Pattern = re.compile(r'''
        ^(25[0-5]|2[0-4]\d|1\d\d|\d\d?) # R (0 - 255)
        \s*,\s*                         # optional spaces, followed by comma, followed by optional spaces.
        (25[0-5]|2[0-4]\d|1\d\d|\d\d?)  # G (0 - 255)
        \s*,\s*                         # optional spaces, followed by comma, followed by optional spaces.
        (25[0-5]|2[0-4]\d|1\d\d|\d\d?)  # B (0 - 255)
        \s*,?\s*                        # optional spaces, followed by optional comma, followed by optional spaces.
        (0|0\.\d{1,2}|1)?$              # A (0 - 1, optional)
        ''', re.VERBOSE)
        self.boost_regex: re.Pattern = re.compile(r'^(\d+\.?\d*)\s*,\s*(\d+\.?\d*)$') # "float, float" with any spaces between floats and comma.
        
        # Default values in the case of initialization without a provided or incomplete `options` dictionary.

        # GIF/Video framerate (frames per second). For GIFs, maximum is 30.
        self.framerate: int = 60
        # Window size for fft calculation (smaller -> more accurate).
        self.fft_window_sec: float = 0.25
        # Amount of bars showing on graph.
        self.bars: int = 20
        # Amount of parts to split each bar to.
        self.bar_parts: int = 0
        # Gap percentage between bar parts.
        self.part_gap: float = 0.2
        # Relative amplitude needed for a frequency to show in the graph. Used to calculate the edges of the graph (between 0 and 1).
        self.amplitude_threshold: float = 0.2
        # (a,b) Boost low amplitude frequencies with the formula: Y = log(a*X+b)/log(a+b).
        self.low_boost: tuple[float] = (0,0)
        # Maximum frames per GIF. Due to high memory usage, please select according to your RAM size and framerate.
        self.max_frames_per_gif: int = 1000
        # Represents image quality (dots per inch).
        self.dpi: int = 100
        # Image sixe in pixels (width, height).        
        self.image_size_pix: Iterable[int,int] = (1920, 1080)
        # Figure backround colour as a string with format: "r,g,b,a"
        self.background: Iterable[float] = (0, 0, 0, 0)
        # Additional options for FFMPEG
        self.ffmpeg_options: list[str]|None = None
        # Colour for bottom of bar gradient
        self.bar_colour_bottom: Iterable[float] = (0, 0, 1)
        # Colour for top of bar gradient
        self.bar_colour_top: Iterable[float] = (1, 0, 0)
        # If specified, exports the bar graph data over time in json format instead of producing an animation file.
        self.export_json: bool = False

        if "framerate" in options.keys():
            self.framerate = options["framerate"]
        if "fft_window_sec" in options.keys():
            if options["fft_window_sec"] < 0:
                raise ValueError("Error: invalid value for fft window, please select a positive value or zero.")
            self.fft_window_sec = options["fft_window_sec"]
        if "bars" in options.keys():
            if options["bars"] <= 0:
                raise ValueError("Error: invalid value for bars, please select a positive value.")
            self.bars = options["bars"]
        if "bar_parts" in options.keys():
            if options["bar_parts"] < 0:
                raise ValueError("Error: invalid value for bar parts, please select a positive value.")
            self.bar_parts = options["bar_parts"]
        if "part_gap" in options.keys():
            if options["part_gap"] < 0 or options["part_gap"] >= 1:
                raise ValueError("Error: invalid value for bar part gap, please select a value between 0 and 1, excluding 1.")
            self.part_gap = options["part_gap"]
        if "amplitude_threshold" in options.keys():
            if options["amplitude_threshold"] < 0 or options["amplitude_threshold"] >= 1:
                raise ValueError("Error: invalid value for amplitude threshold, please select a value between 0 and 1.")
            self.amplitude_threshold = options["amplitude_threshold"]
        if "low_boost" in options.keys():
            boost_match: re.Match = re.match(self.boost_regex, options["low_boost"].strip())
            if boost_match is None:
                raise ValueError("Error: invalid low frequency boost input. Please provide in format `a,b`.")
            a, b = boost_match.groups()
            a, b = float(a), float(b)
            if (a == 0) != (b == 0) or 0 < a < 1 or 0 < b < 1:
                raise ValueError("Error: invalid low frequency boost input, a and b must be greater or equal to 1, or both 0.")
            self.low_boost = (a,b)
        if "max_frames_per_gif" in options.keys():
            self.max_frames_per_gif = options["max_frames_per_gif"]
        if "dpi" in options.keys():
            if options["dpi"] <= 0:
                raise ValueError("Error: invalid value for dpi, please provide a positive value.")
            self.dpi = options["dpi"]
        if "width" in options.keys() and "height" in options.keys():
            if options["width"] <= 0:
                raise ValueError("Error: Invalid value for width, please provide a positive value.")
            if options["height"] <= 0:
                raise ValueError("Error: Invalid value for height, please provide a positive value.")
            self.image_size_pix = (options["width"], options["height"])
        if "background" in options.keys():
            bg_match: re.Match = re.match(self.rgba_regex, options["background"].strip())
            if bg_match is None:
                raise ValueError("Error: invalid backround colour input, please provide in the format `r,g,b,a`.")
            rgba: tuple[str|None] = bg_match.groups()
            if rgba[-1] is None:
                rgba = rgba[:-1]
                self.background = tuple([int(i) / 255 for i in rgba] + [1])
            else:
                self.background = tuple([int(i) / 255 for i in rgba[:-1]] + [float(rgba[-1])])
        if "ffmpeg_options" in options.keys():
            if not options["ffmpeg_options"]:
                self.ffmpeg_options = None
            else:
                self.ffmpeg_options = options["ffmpeg_options"].split()
                self.ffmpeg_options = [option.strip() for option in self.ffmpeg_options]
        if "bar_colour_bottom" in options.keys():
            bar_bottom_match: re.Match = re.match(self.rgba_regex, options["bar_colour_bottom"].strip())
            if bar_bottom_match is None:
                raise ValueError("Error: invalid `bar_colour_bottom` input, please provide in the format `r,g,b`.")
            rgb: tuple[str] = bar_bottom_match.groups()[:3]
            self.bar_colour_bottom = tuple([int(i) / 255 for i in rgb])
        if "bar_colour_top" in options.keys():
            bar_top_match: re.Match = re.match(self.rgba_regex, options["bar_colour_top"].strip())
            if bar_top_match is None:
                raise ValueError("Error: invalid `bar_colour_top` input, please provide in the format `r,g,b`.")
            rgb: tuple[str] = bar_top_match.groups()[:3]
            self.bar_colour_top = tuple([int(i) / 255 for i in rgb])
        if "export_json" in options.keys():
            self.export_json = options["export_json"]
