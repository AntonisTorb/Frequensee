from __future__ import annotations
import argparse
from pathlib import Path
import sys

from .audio_viz import AudioVisualizer
from .boost_info import test_boost
from .config import Config
from .version import __version__


def main_cli():
    '''Function to be called with the `frequensee` or `fqc` cli command created when installing as package.'''

    print(f'Frequensee v{__version__}. Use command `fqc -h` for a list of commands.')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=False,
                        help="Filepath to the audio file.")
    parser.add_argument("-r", "--framerate", type=int, required=False, default=60, 
                        help="Animation framerate (frames per second, default: 60). For GIFs maximum is 30, adjusted automatically.")
    parser.add_argument("-fw", "--fft_window_sec", type=float, required=False, default=0.25, 
                        help="Window size for fft calculation (smaller -> more accurate, positive, default: 0.25).")
    parser.add_argument("-b", "--bars", type=int, required=False, default=20, 
                        help="Amount of bars showing on graph (non zero positive, default: 20).")
    parser.add_argument("-bp", "--bar_parts", type=int, required=False, default=0,
                        help="Amount of parts to split each bar to, with 0 being a gradient (Positive integer or 0, default: 0).")
    parser.add_argument("-pg", "--part_gap", type=float, required=False, default=0.2,
                        help="Gap between bar parts as a percentage of the bar length (Between 0 and 1, excluding 1, default: 0.1).")
    parser.add_argument("-t", "--amplitude_threshold", type=float, required=False, default=0.2, 
                        help="Minimum relative amplitude for frequencies, used to calculate the edges of the graph (between 0 and 1, default: 0.2).")
    parser.add_argument("-lb", "--low_boost", type=str, required=False, default="0,0", 
                        help='''Boost low amplitude frequencies with the formula: Y = log(a*X+b)/log(a+b).
    Provided in the format "a,b", with a,b: floats greater or equal to 1, or both zero for no boost (default: "0,0")''')
    parser.add_argument("-tb", "--test_boost", action="store_true", required=False,
                        help='''Create a graph showing the output of the boost function with the provided parameters. 
                        Required to pass the `a,b` parameters with the `-lb` option.''')
    parser.add_argument("-g", "--max_frames_per_gif", type=int, required=False, default=1000, 
                        help="Maximum frames per GIF. Due to high memory usage, please select according to your RAM size and framerate (Default: 1000).")
    parser.add_argument("-d", "--dpi", type=int, required=False, default=100, 
                        help="Represents image quality (dots per inch, default: 100).")
    parser.add_argument("-w", "--width", type=int, required=False, default=1080, 
                        help="Width of resulting animation in pixels (Default: 1080).")
    parser.add_argument("-ht", "--height", type=str, default=1920, 
                        help="Height of resulting animation in pixels (Default: 1920).")
    parser.add_argument("-bg", "--background", type=str, required=False, default="0,0,0,0", 
                        help='''Figure backround colour as a string with format: 'red,green,blue,alpha', alpha is optional. 
    Red/Green/Blue values: Between 0 and 255. Alpha value between 0 and 1. (Default: '0,0,0,0')''')
    parser.add_argument("-bb", "--bar_colour_bottom", "--bar_color_bottom", type=str, required=False, default="0,0,255", 
                        help='''RGB colour for the bottom of the bar gradient in the format `red,green,blue`.
    Red/Green/Blue values: Between 0 and 255. (Default: `0,0,255`)''')
    parser.add_argument("-bt", "--bar_colour_top", "--bar_color_top", type=str, required=False, default="255,0,0", 
                        help='''RGB colour for the top of the bar gradient in the format `red,green,blue`.
    Red/Green/Blue values: Between 0 and 255. (Default: `255,0,0`)''')
    parser.add_argument("-f", "--ffmpeg_options", type=str, required=False, default="", 
                        help="Additional options for FFMPEG as a string separated by space. Do not include spaces in the arguments.")
    parser.add_argument("-j", "--export_json", action="store_true", required=False, 
                        help='''If specified, exports the bar graph data over time in json format instead of producing an animation file.
    Includes audio filepath and framerate for which the data was created.''')
    parser.add_argument("-fft", "--animate_fft", action="store_true", required=False, 
                        help="If specified, creates an animation of the raw fft over time instead of the bars.")
    parser.add_argument("-o", "--output_path", type=str, required=False,
                        help="Path or filename of output file (including extension compatible with FFMPEG or json).")
    
    args = parser.parse_args()
    
    options: dict[str, str|int|float|bool|None] = {   
        "framerate": args.framerate,
        "fft_window_sec": args.fft_window_sec,
        "bars": args.bars,
        "bar_parts": args.bar_parts,
        "part_gap": args.part_gap,
        "amplitude_threshold": args.amplitude_threshold,
        "low_boost": args.low_boost,
        "max_frames_per_gif": args.max_frames_per_gif,
        "dpi": args.dpi,
        "width": args.width,
        "background": args.background,
        "ffmpeg_options": args.ffmpeg_options,
        "bar_colour_bottom": args.bar_colour_bottom,
        "bar_colour_top": args.bar_colour_top,
        "export_json": args.export_json
    }

    config = Config(options)

    if args.test_boost:
        if config.low_boost == (0,0):
            raise ValueError("Error: Please provide valid parameters for the boost.")
        test_boost(*config.low_boost)
        sys.exit()

    if not args.input_path:
        raise ValueError("Error: Input path is required.")
    if not args.output_path:
        raise ValueError("Error: Output path is required.")
            
    input_path: Path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError("Error: Provided filepath does not exist.")
    
    output_path: Path = Path(args.output_path)
    if "." not in output_path.name:
        raise ValueError("Error: output filepath must end with filename including extension.")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    viz = AudioVisualizer(config)
    viz.load(input_path)

    if args.export_json:
        if output_path.suffix != ".json":
            raise ValueError("Error: Output file must have `.json` suffix.")
        viz.extract_json(output_path)
    elif args.animate_fft:
        viz.create_fft_animation(output_path)
    else:
        viz.create_bar_animation(output_path)
