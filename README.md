# Frequensee

A command line interface (`cli`) tool used to create animations based on audio frequencies by utilizing the `Fast Fourier Transform (fft)` of the audio data. `Frequensee` uses `FFMPEG` to create the animations, so it either needs to be installed or have the `ffmpeg.exe` executable be present in the working directory.

Bar animation|
:-------------------------:
![image](https://raw.githubusercontent.com/AntonisTorb/Frequensee/refs/heads/main/images/example_bars.webp) 

FFT animation|
:-------------------------:
![image](https://raw.githubusercontent.com/AntonisTorb/Frequensee/refs/heads/main/images/example_fft.webp)

## Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [How to use](#how-to-use)
    - [Cli version](#cli-version)
    - [Importing from source](#importing-from-source)
- [Supported File formats](#supported-file-formats)
    - [Input](#input)
    - [Output](#output)
- [Common output formats: Advantages and Disadvantages](#common-output-formats-advantages-and-disadvantages)
- [Command line options](#command-line-options)

## Installation

`Frequensee` is released as a package on [PyPI](https://pypi.org/project/frequensee/), so you can install it with pip using the following command:

```
pip install frequensee
```

or by downloading the source code from the [GitHub repository](https://github.com/AntonisTorb/Frequensee).

## Dependencies

`Frequensee` has 3 major dependencies:

- [Soundfile](https://pypi.org/project/soundfile/): Used to read the data of the audio file and return them as a `numpy` array.
- [Matplotlib](https://pypi.org/project/matplotlib/): Used to create the frames of the animation basd on the data above.
- [FFMPEG](https://www.ffmpeg.org/): Used to combine the frames and produce the animation file.

The first two are installed automatically when installing `frequensee` through `PyPI`. You can also install them with pip using the `requirements.txt` file provided.

As for `FFMPEG`, you can either compile the source code yourself, or use an already comppiled version. Personally, I use the executable provided by [yt-dlp](https://github.com/yt-dlp/FFmpeg-Builds/releases/tag/latest). The installation process is not too straight forward, but there are several available tutorials online. IN any case, you can also have the `FFMPEG` executable file in the current working directory and it should work as well.

## How to use

### Cli version

If you installed through `PyPI`, the easiest way to run `frequensee` is with either of the following commands:
```
fqc -i "input file path" -o "output file path"
frequensee -i "input file path" -o "output file path"
```
Examples:
```
fqc -i "input.wav" -o "videos/output.gif"
frequensee -i "audio/input.mp3" -o "output.mp4"
```

If you installed from source, you can run the `main.py` file instead (remember to install the dependancies in `requirements.txt` first):
```
python main.py -i "input file path" -o "output file path"
```
or copy the code inside the `main.py` file to your desired python file.

Additional command line options can be found in the [command line options](#command-line-options) section.

### Importing from source

Instead of running the `cli` version, it is possible to import and create all the components of `frequensee` and customize them from the source code. For example:

```
from frequensee.config import Config
from frequensee.audio_viz import AudioVisualizer

config = Config()
viz = AudioVisualizer(config)
viz.load("path to audio file)
viz.create_bar_animation("path to output file")

viz.load("path to other audio file)
config2 = Config(different options)
viz.load_config(config2)

viz.create_fft_animation("path to output file")
viz.extract_json("path to output file")
```

The `Config` class contains values required for the `AudioVisualizer` instance to create the animations/export the data. Please refer to the class documentation if you wish to configure the default values.

The `load` method will read the audio data from the audio file in the provided filepath.

The `load_config` method can be used to update the configuration of the `AudioVisualizer` instance after it has been initialized.

There are 2 main `AudioVisualizer` methods used to produce the animations:

- `create_bar_animation`: Creates an animatioin of bar shaped gradient images based on the relative amplitude. The frequency range is not visible in the resulting animation and the result is purely aesthetic.

- `create_fft_animation`: Creates an animation of the relative amplitude over frequency, over time, as it was created from the Fast Fourier Transform (fft) of the input audio data. 

Additionally, there is the option to export the bar amplitude data in `json` format, if you prefer to use a different visualization methods or tool, by using the `extract_json` method. The structure is as follows:
```
{
    "audio_filepath": "path to input audio file (str)",
    "bars": "amount of bars created for the visuals (int)",
    "bar_graph": [
        [array of relative bar amplitudes for first frame (floats)],
        ...
        [array of relative bar amplitudes for last frame (floats)]
    ]
}
```

The `bar_graph` array's length is equal to the number of frames generated, while each frame array's length is equal to the number of bars.

## Supported file formats

### Input
While it has not been tested, all formats compatible with the python package `soundfile` should be compatible with `frequensee`.

    AIFF: AIFF (Apple/SGI)
    AU: AU (Sun/NeXT)
    AVR: AVR (Audio Visual Research)
    CAF: CAF (Apple Core Audio File)
    FLAC: FLAC (Free Lossless Audio Codec)
    HTK: HTK (HMM Tool Kit)
    SVX: IFF (Amiga IFF/SVX8/SV16)
    MAT4: MAT4 (GNU Octave 2.0 / Matlab 4.2)
    MAT5: MAT5 (GNU Octave 2.1 / Matlab 5.0)
    MPC2K: MPC (Akai MPC 2k)
    MP3: MPEG-1/2 Audio
    OGG: OGG (OGG Container format)
    PAF: PAF (Ensoniq PARIS)
    PVF: PVF (Portable Voice Format)
    RAW: RAW (header-less)
    RF64: RF64 (RIFF 64)
    SD2: SD2 (Sound Designer II)
    SDS: SDS (Midi Sample Dump Standard)
    IRCAM: SF (Berkeley/IRCAM/CARL)
    VOC: VOC (Creative Labs)
    W64: W64 (SoundFoundry WAVE 64)
    WAV: WAV (Microsoft)
    NIST: WAV (NIST Sphere)
    WAVEX: WAVEX (Microsoft)
    WVE: WVE (Psion Series 3)
    XI: XI (FastTracker 2)

### Output

While it has not been tested, all formats compatible with your version of `FFMPEG` should be compatible with `frequensee`.

You can check the formats and codecs supported in your version of `FFMPEG` by running the command:
```
ffmpeg -formats
ffmpeg -codecs
```

Some formats might require additional ffmpeg options that can be passed by using the [`-f`command line option](#command-line-options) to the `cli` version.

Unfortunately, the order of `FFMPEG` parameters matters, so it might be difficult to use the `cli` version of `frequensee` for some formats. In such cases, please look into the `writers.py` module, which contains a custom `FFMpegWriter` class. It's possible to add options in the `_args` method in the correct place in the command.

## Common output formats: Advantages and Disadvantages
- `mp4`: The creation of mp4 files is the default configuration for video formats. The advantages include low memory usage, embedded audio, and great compatibilty. 

    On the other hand, the biggest disadvantage is the lack of background transparency. Of course, with a proper choice of background color, it will be possible to introduce transparency with video editing software.

- `gif`: The biggest advantage of gifs is the option to have a transparent background on the resulting animation without having to use additional editing software.

    This comes with many disadvantages however. Memory usage for gif creation with `FFMPEG` rises rapidly and can easily fill up your system's available memory. Additionally, the maximum framerate is limited to `30fps`, cannot include the source audio, and the resulting filesize can be orders of magnitude larger than an mp4 file.

- `webp`: A format similar to `gif`, with the added advantages of smaller file size and low memory usage during creation. Unfortunately, it has many compatibility issues even in modern systems and is mostly useful in web development.

For now, only `gif` and `webp` are natively supported for image formats.

For the above reasons, it is recommended to use `mp4` or `webp` as the output format if possible. If a `gif` is needed, please make sure to limit the amount of frames included in each resulting `gif part` with the `-g` [command line option](#command-line-options), taking into account the input audio length, the available memory of your system, as well as the resulting framerate.


## Command line options

You can get the following overview by using the help cli flag:
```
fqc -h
```

    -h, --help
        Show this help message and exit.

    -i INPUT_PATH, --input_path INPUT_PATH
        Filepath to the audio file.

    -r FRAMERATE, --framerate FRAMERATE
        Animation framerate (frames per second, default: 60). For GIFs maximum is 30, adjusted automatically.

    -fw FFT_WINDOW_SEC, --fft_window_sec FFT_WINDOW_SEC
        Window size for fft calculation (smaller -> more accurate, default: 0.25).

    -b BARS, --bars BARS 
        Amount of bars showing on graph (Default: 20).

    -bp BAR_PARTS, --bar_parts BAR_PARTS
        Amount of parts to split each bar to, with 0 being a gradient (Positive integer or 0, default: 0).

    -pg PART_GAP, --part_gap PART_GAP
        Gap between bar parts as a percentage of the bar length (Between 0 and 1, excluding 1, default: 0.2).

    -t AMPLITUDE_THRESHOLD, --amplitude_threshold AMPLITUDE_THRESHOLD
        Minimum relative amplitude for frequencies, used to calculate the edges of the graph (between 0 and 1, default: 0.2).

    -g MAX_FRAMES_PER_GIF, --max_frames_per_gif MAX_FRAMES_PER_GIF
        Maximum frames per GIF. Due to high memory usage, please select according to your RAM size and framerate (Default: 1000).

    -d DPI, --dpi DPI
        Represents image quality (dots per inch, default: 100).

    -w WIDTH, --width WIDTH
        Width of resulting animation in pixels (Default: 1080).

    -ht HEIGHT, --height HEIGHT
        Height of resulting animation in pixels (Default: 1920).

    -bg BACKGROUND, --background BACKGROUND
        Figure backround colour as a string with format: 'red,green,blue,alpha', alpha is optional. Red/Green/Blue values: Between 0 and 255. Alpha value between 0 and 1. (Default: '0,0,0,0')

    -bb BAR_COLOUR_BOTTOM, --bar_colour_bottom BAR_COLOUR_BOTTOM, --bar_color_bottom BAR_COLOUR_BOTTOM
        RGB colour for the bottom of the bar gradient in the format `red,green,blue`. Red/Green/Blue values: Between 0 and 255. (Default: `0,0,255`)

    -bt BAR_COLOUR_TOP, --bar_colour_top BAR_COLOUR_TOP, --bar_color_top BAR_COLOUR_TOP
        RGB colour for the top of the bar gradient in the format `red,green,blue`. Red/Green/Blue values: Between 0 and 255. (Default: `255,0,0`)

    -f FFMPEG_OPTIONS, --ffmpeg_options FFMPEG_OPTIONS
        Additional options for FFMPEG as a string separated by space. Do not include spaces in the arguments.

    -j, --export_json     
        If specified, exports the bar graph data over time in json format instead of producing an animation file. Includes audio filepath and framerate for which the data was created.

    -fft, --animate_fft   
        If specified, creates an animation of the raw fft over time instead of the bars.

    -o OUTPUT_PATH, --output_path OUTPUT_PATH
        Path or filename of output file (including extension compatible with FFMPEG or json).

The only required flags are the input and output filepaths.


## TODO

- Modify workflow for `test.pypi` release to only run when there is a package version update.
- Documentation.
- Better exception handling.
