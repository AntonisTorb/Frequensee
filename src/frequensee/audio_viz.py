from __future__ import annotations
from collections.abc import Callable
from functools import partial
import json
from math import ceil
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import soundfile as sf

from .config import Config
from .writers import FFMpegWriterWithAudio


class AudioVisualizer():

    def __init__(self, config: Config) -> None:
        '''
        Audio frequency visualizer using the `Fast Fourier Transform (fft)` in order to produce animations with FFMPEG.
        Methods (Please check the documentation of each or the `README` for additional information):
        - `load()`: Load the audio file data>
        - `load_config`: Load a new configuration after initialization.
        - `create_bar_animation`: Create a bar animation with the audio fft data.
        - `create_fft_animation`: Create an fft animation with the audio fft data.
        - `extract_json`: Extract the bar data in `json` format

        Parameters
        ----------
        config : `config.Config`
            Instance of `Config` class containing configuration options (check `config.py`).
        '''

        self.config = config

        # If FFMPEG is not installed or `ffmpeg.exe` not in working directory, 
        # raise error, unless exporting json.
        if ((shutil.which("ffmpeg") is None and 
            shutil.which(Path.cwd() / "ffmpeg.exe") is None) and 
            not self.config.export_json):
            raise FileNotFoundError("Error: FFMPEG is not installed.")


    def _calculate_audio_fft_properties(self) -> None:
        '''Calculates and saves required properties for the fft calculations.'''
        
        self.fft_window_size = int(self.sample_rate * self.config.fft_window_sec)
        self.audio_length_sec: float = self.audio.shape[0] / self.sample_rate
        self.fft_frequency_array: np.ndarray = np.fft.rfftfreq(self.fft_window_size, 1/self.sample_rate)


    def load(self, audio_path: Path|str) -> None:
        '''
        Loads the audio file data and performs calculations for required properties.
        
        Parameters
        ----------
        audio_path : `pathlib.Path | str`
            Path to the input audio file.
        '''

        self.audio_path = Path(audio_path)
        time_series: np.ndarray
        time_series, self.sample_rate = sf.read(audio_path, always_2d=True)
        self.audio: np.ndarray = time_series.T[0]

        self._calculate_audio_fft_properties()


    def load_config(self, config: Config) -> None:
        '''
        Loads configuration instance after `AudioVisualizer` had been initialized.
        Re-performs calculations based on new configuration.

        
        Parameters
        ----------
        config : `config.Config`
            Instance of `Config` class containing configuration options (check `config.py`).
        '''
    
        self.config = config
        self._calculate_audio_fft_properties()


    def _extract_sample(self, frame_number: int) -> np.ndarray:
        '''
        Extracts audio sample based on the provided frame number and window in configuration 
        and returns it for the fft calculation.
        
        Parameters
        ----------
        frame_number : `int`
            Frame number for which to calculate the fft array.

        Returns
        -------
        `np.ndarray`
            Numpy array containing the data in the window.
        '''

        frame_end = frame_number * self.frame_offset
        frame_start = int(frame_end - self.fft_window_size)

        if frame_end == 0:
            return np.zeros((np.abs(frame_start)), dtype=float)
        elif frame_start < 0:
            return np.concatenate([np.zeros((np.abs(frame_start)), dtype=float), self.audio[0:frame_end]])
        else:
            return self.audio[frame_start:frame_end]


    def _create_fft_array(self) -> np.ndarray:
        '''
        Calculates and returns a normalized array containing the fft data of all samples.
        
        Returns
        -------
        `np.ndarray`
            Numpy array containing the fft data for all frames normalized in range [0,1].
        '''

        print("Creating fft frames...")
        fft_array = []
        for frame_number in range(self.frame_count):
            if self.frame_count > 1000:
                print(f'{(100*frame_number/self.frame_count):.2f}%', end="\r")

            sample = self._extract_sample(frame_number)
            fft = np.fft.rfft(sample)
            fft = np.abs(fft).real
            fft_array.append(fft)

        print(f'Done.{" "*20}')

        fft_array = np.array(fft_array)
        return fft_array / fft_array.max()


    def _get_frequency_ranges(self, fft_array: np.ndarray) -> tuple[int]:
        '''Determines and returns the starting and ending indexes of the array between which 
        the amplitude is sufficiently high based on the configuration.
        
        Parameters
        ----------
        fft_array : `np.ndarray`
            Fft array created by the `_create_fft_array` method.

        Returns
        -------
        `tuple[int, int]`
            The start and end column indexes of the `fft_array` defining the range to be visualized.
        '''

        max_of_cols: np.ndarray = fft_array.max(axis=0)
        start = 0
        end = 0
        for index, max in enumerate(max_of_cols):
            if max >= self.config.amplitude_threshold:
                start = index
                break

        for index, max in enumerate(reversed(max_of_cols)):
            if max >= self.config.amplitude_threshold:
                end = len(max_of_cols) - index
                break

        # print(start, end)
        # print(self.fft_frequency_array[start], self.fft_frequency_array[end])
        return start, end


    def _create_bar_array(self, fft_array: np.ndarray, start: int, end: int) -> np.ndarray:
        '''
        Creates and returns a numpy array containing all the values for the bar visualization.
        The array is split into parts according to the provided amount of bars and the sum of 
        the values is used for the bar height. The array is normalized based on the maximum value.

        Parameters
        ----------
        fft_array : `np.ndarray`
            Fft array created by the `_create_fft_array` method.
        start : `int`
            Start index for `fft_array`.
        end : `int`
            End index for `fft_array`.

        Returns
        -------
        `np.ndarray`
            Numpy array containing the values for the bar visualization.
        '''

        # Boosting lower frequencies if configured.
        if self.config.low_boost != (0,0):
            a = self.config.low_boost[0]
            b = self.config.low_boost[1]
            fft_array = np.apply_along_axis(lambda x: np.log(a*x + b) / np.log(a + b), 1, fft_array)

        bar_array = []
        for frame in fft_array:
            frame: np.ndarray = frame[start:end]
            chunks: list[np.ndarray] = np.array_split(frame, self.config.bars)
            bar_array.append([chunk.sum() for chunk in chunks])
        bar_array = np.array(bar_array)

        return bar_array / bar_array.max()


    def _get_colour_gradient_full(self, h: float) -> np.ndarray:
        '''
        Used by the `_create_bar_graph_frame` method to display a continuous gradient bar image. 
        Creates and returns a numpy array with the RGB values for the image gradient.

        Parameters
        ----------
        h : `float`
            The height of the bar.

        Returns
        -------
        `np.ndarray`
            Numpy array of RGB values for the gradient with shape `[self.config.dpi, 1, 3]`.
        '''

        r = np.linspace([self.config.bar_colour_top[0] * h + self.config.bar_colour_bottom[0] * (1 - h)],
                [self.config.bar_colour_bottom[0]],
                self.config.dpi)[:, :, None]
        g = np.linspace([self.config.bar_colour_top[1] * h + self.config.bar_colour_bottom[1] * (1 - h)],
                [self.config.bar_colour_bottom[1]],
                self.config.dpi)[:, :, None]
        b = np.linspace([self.config.bar_colour_top[2] * h + self.config.bar_colour_bottom[2] * (1 - h)],
                [self.config.bar_colour_bottom[2]],
                self.config.dpi)[:, :, None]
        
        grad = np.concatenate([r, g, b], axis=2)
        # print(grad.shape)
        return grad


    def _get_colour_gradient_parts(self, h: float, alpha_top: float) -> np.ndarray:
        '''
        Used by the `_create_bar_graph_frame` method to display a gradient bar image split in parts.
        Creates and returns a numpy array with the RGBA values for the image gradient.

        Parameters
        ----------
        h : `float`
            The height of the bar.

        Returns
        -------
        `np.ndarray`
            Numpy array of RGBA values for the gradient with shape:
            `[(int(self.config.bar_parts * h) * self.loop_range , 1, 4]`.
        '''

        # `np.linspace`` doesn't work here since we want non-continuous values. Is there an alternative with np?

        # RGB calculation: the second loop is needed to create a percentage based alpha channel. Need to maintain dimensions.
        r = [[[
                (self.config.bar_colour_bottom[0] + 
                (self.config.bar_colour_top[0] - self.config.bar_colour_bottom[0]) * 
                i / self.config.bar_parts)
            ]] for i in reversed(range(int(self.config.bar_parts * h))) for _ in range(self.loop_range)
        ]
        g = [[[
                (self.config.bar_colour_bottom[1] + 
                (self.config.bar_colour_top[1] - self.config.bar_colour_bottom[1]) *
                i / self.config.bar_parts)
            ]] for i in reversed(range(int(self.config.bar_parts * h))) for _ in range(self.loop_range)
        ]
        b = [[[
                    (self.config.bar_colour_bottom[2] + 
                    (self.config.bar_colour_top[2] - self.config.bar_colour_bottom[2]) * 
                    i / self.config.bar_parts)
            ]] for i in reversed(range(int(self.config.bar_parts * h))) for _ in range(self.loop_range)
        ]
        # Alpha calculation: 1 only if i between lower and upper limits, else 0.
        # For top bar part, opacity based on height.
        a = ([
            [
                [int((i > self.alpha_lower_limit) and (i < self.alpha_upper_limit)) * alpha_top]
            ] for i in range(self.loop_range)
        ] + 
        [
            [
                [int((i > self.alpha_lower_limit) and (i < self.alpha_upper_limit))]
            ] for _ in reversed(range(int(self.config.bar_parts * h - 1))) for i in range(self.loop_range)
        ])

        grad = np.concatenate([r, g, b, a], axis=2)
        # print(grad.shape)
        return grad


    def _create_fft_frame(self, frame_number: int, artists: list[Line2D], 
                          y_values: np.ndarray) -> list[Line2D]:
        '''
        Used by the `_create_animation_file` method. 
        Updates the data of the artists for each frame and returns the list of artists.

        Parameters
        ----------
        frame_number : `int`
            The frame number.
        artists: list[matplotlib.lines.Line2D]
            List of `Line2D` objects to update in the animation.
        y_values : `np.ndarray`
            Numpy array containing the values for the Y axis (amplitude arrays for each frame).
            

        Returns
        -------
        `list[matplotlib.lines.Line2D]`
            List of updated `Line2D` objects.
        '''

        for line in artists:
            line.set_data(self.fft_frequency_array, y_values[frame_number])

        print(f'{(100*frame_number/self.frame_count):.2f}%', end="\r")
        return artists


    def _create_bar_graph_frame(self, frame_number: int, artists: list[AxesImage], 
                                y_values: np.ndarray) -> list[AxesImage]:
        '''
        Used by the `_create_animation_file` method. 
        Updates the data and extent of the artists for each frame and returns the list of artists.

        Parameters
        ----------
        frame_number : `int`
            The frame number.
        artists: list[matplotlib.image.AxesImage]
            List of `AxesImage` objects to update in the animation.
        y_values : `np.ndarray`
            Numpy array containing the values for the Y axis (amplitude arrays for each frame).
            
        Returns
        -------
        `list[matplotlib.image.AxesImage]`
            List of updated `AxesImage` objects.
        '''

        for bar_coords, img, h in zip(self.coords, artists, y_values[frame_number]):
            if self.config.bar_parts:
                # Minimum 1 bar part.
                if h <= 1 / self.config.bar_parts:
                    h = 1 / self.config.bar_parts
                
                # Set top bar part alpha based on height.
                alpha_top = (h % (1 / self.config.bar_parts)) * self.config.bar_parts
                if alpha_top < 0.01:
                    alpha_top = 1

                # Normalize height so bar parts all have the same height.
                h = ceil(self.config.bar_parts * h) / self.config.bar_parts

                grad = self._get_colour_gradient_parts(h, alpha_top)
            else:
                grad = self._get_colour_gradient_full(h)
            img.set_data(grad)
            img.set_extent([bar_coords[0], bar_coords[0] + bar_coords[1], bar_coords[2], bar_coords[2] + h])

        print(f'{(100*frame_number/self.frame_count):.2f}%', end="\r")
        return artists


    def _create_writer(self, filepath) -> FFMpegWriter|FFMpegWriterWithAudio:
        '''
        Creates and returns the writer object for the `_create_animation_file` method in order to save the animation file.

        Parameters
        ----------
        filepath : `Path | str`
            The filepath to the output file.

        Returns
        -------
        `FFMpegWriter` or `FFMpegWriterWithAudio`
            Appropriate writer initialized with options according to the output file format.
        '''

        # from matplotlib.animation import PillowWriter
        # return PillowWriter(fps=self.config.framerate)

        if Path(filepath).suffix not in [".gif", ".webp"]:
            return FFMpegWriterWithAudio(fps=self.config.framerate, audio_filepath=self.audio_path,
                                            extra_args=self.config.ffmpeg_options)
        
        if Path(filepath).suffix == ".webp":
            if self.config.ffmpeg_options is not None:
                self.config.ffmpeg_options = (
                    ["-c:v", "webp", "-loop", "0", "-pix_fmt", "yuva420p"] + self.config.ffmpeg_options
                )
            else:
                self.config.ffmpeg_options = ["-c:v", "webp", "-loop", "0", "-pix_fmt", "yuva420p"]

        return FFMpegWriter(fps=self.config.framerate, extra_args=self.config.ffmpeg_options)


    def _initial_frame(self, ax: Axes, x_values: list[int]|np.ndarray, 
                       animation_func: Callable) -> list[Artist]:
        '''
        Creates the initial animation frame and returns the resulting list of `Artist` objects.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            The matplotlib axes where the frame is drawn.
        x_values: `list[int] | np.ndarray`
            The values for the x axis. List of integers for bar animation, numpy array for fft animation.
        animation_func: `collections.abc.Callable`
            Function to be used by `FuncAnimation` to produce the animation frames.

        Returns
        -------
        `list[matplotlib.artist.Artist]`
            List of `Artist` object resulting from the initial frame drawing. 
            Artists are `matplotlib.image.AxesImage` for bar animation and `matplotlib.lines.Line2D` for fft animation.
        '''

        ax.clear()
        ax.set_ylim(0,1)
        ax.patch.set_alpha(0)

        if animation_func == self._create_fft_frame:
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Relative Amplitude")
            ax.patch.set_alpha(0)
            ax.set_xlim(x_values[0], x_values[-1])

            artists: list[Line2D] = ax.plot(x_values, [0 for _ in x_values])
            return artists


        plt.axis("off")
        plt.margins(x=0)
        lim = ax.get_xlim() + ax.get_ylim()
        ax.axis(lim)
        
        bar: Rectangle
        bars = ax.bar(x_values, [1 for _ in x_values])

        artists: list[AxesImage] = []
        grad_init: np.ndarray = np.array([[[0,0,0,0]]])

        # Keeping track of bar coordinates to update in animation.
        self.coords: list[tuple[float]] = []

        # Replace bars with color gradient image based on bar height.
        for bar in bars:
            bar.set_alpha(0)

            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            self.coords.append((x,w,y))

            img = ax.imshow(grad_init, extent=[x, x + w, y, y + h], aspect="auto", zorder=10)
            artists.append(img)
        ax.set_xlim(1 - w/2, w/2 + self.config.bars)

        return artists


    def _create_animation_file(self, x_values: list|np.ndarray, y_values: np.ndarray, 
                               animation_func: Callable, filepath: Path|str) -> None:
        '''
        Creates the animation file(s) in the specified path.

        Parameters
        ----------
        x_values : `np.ndarray`
            Numpy array containing the values for the X axis.
        y_values : `np.ndarray`
            Numpy array containing the values for the Y axis for each frame.
        animation_func: `collections.abc.Callable`
            Function to be used by `FuncAnimation` to produce the animation frames.
        filepath: `Path | str`
            The filepath where the resulting animation file(s) will be saved (must contain file extension).
        '''

        self.image_size_inch = tuple([(size_pixel / self.config.dpi) for size_pixel in self.config.image_size_pix])
        fig, ax = plt.subplots(figsize=self.image_size_inch)
        fig.patch.set_alpha(self.config.background[-1])
        fig.patch.set_color(self.config.background[:-1])
        if animation_func == self._create_bar_graph_frame:
            fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0) 

        writer = self._create_writer(filepath)

        if self.config.bar_parts:
            # Precalculations to reduce load on every frame creation loop if we need alpha channel for bar parts.
            self.loop_range: int = int(100 * self.config.part_gap)
            self.alpha_lower_limit: float = self.loop_range * self.config.part_gap / 2
            self.alpha_upper_limit: float = self.loop_range - self.alpha_lower_limit
        
        artists: list[Artist] = self._initial_frame(ax, x_values, animation_func)

        if not (Path(filepath).suffix == ".gif" and 
            self.config.max_frames_per_gif and 
            y_values.shape[0] > self.config.max_frames_per_gif):

            print(f'Creating {filepath}...')
            ani = FuncAnimation(fig, 
                                partial(animation_func, artists=artists, y_values=y_values),
                                frames = y_values.shape[0])

            ani.save(filepath, writer=writer, dpi=self.config.dpi,
                    savefig_kwargs={"transparent": True})

            print(f'Done.{" "*20}')
            plt.close()
            return

        # Splitting gifs to parts.
        parts = int(y_values.shape[0] / self.config.max_frames_per_gif) + 1
        y_values_list = np.array_split(y_values, parts)

        for i, y_values in enumerate(y_values_list):

            # Potential issue: if using the gif parts with audio editing software in timelines with higher framerate,
            # the last frame of the gif can be corrupted if the length of the y_values is odd. Tested with DaVinci Resolve, 60FPS.
            # You can trim the last frame manually, but this can get lengthy. Unsure about this behaviour.
            # Fix below works, but it affects the timing of the animation, with higher effect the more parts there are.
            # Recommended to not use for large gifs, use a video file format instead.
            # If you need transparency, set the background to a colour very different from the animation colours and remove in
            # editing software.
            
            # Fix for timelines in video editing software that have a framerate of 60. Issue caused if array rows are odd.
            # if y_values.shape[0] % 2:
            #     y_values = np.append(y_values, y_values[-1:], axis=0)
            
            # print(y_values.shape[0])
            
            filepath_parts = list(Path(filepath).parts)
            filename_parts = filepath_parts[-1].split(".")
            filename_parts[-2] = f'{filename_parts[-2]}-part{i+1}'
            filepath_parts[-1] = ".".join(filename_parts)
            filepath_part = Path(*filepath_parts)

            print(f'Creating {filepath_part}, {i+1}/{len(y_values_list)}...')
            
            ani = FuncAnimation(fig, 
                        partial(animation_func, artists=artists, y_values=y_values), 
                        frames = y_values.shape[0])

            ani.save(filepath_part, writer=writer, dpi=self.config.dpi,
                savefig_kwargs={"transparent": True})
            
        print(f'Done.{" "*20}')
        plt.close()
        

    def extract_json(self, filepath: Path|str) -> None:
        '''
        Extracts the audio filename, ammount of bars and the bar array in a JSON file.
        
        Parameters
        ----------
        filepath : `Path | str`
            The filepath where the resulting `json` file will be saved (must contain `.json` file extension). 
        '''

        self.frame_count = int(self.audio_length_sec * self.config.framerate)
        self.frame_offset = int(len(self.audio)/self.frame_count)

        fft_array = self._create_fft_array()
        start, end = self._get_frequency_ranges(fft_array)
        bar_array: np.ndarray = self._create_bar_array(fft_array, start, end)

        data = {
            "audio_filepath": str(self.audio_path.resolve()),
            "bars": self.config.bars,
            "bar_graph": bar_array.tolist()
        }

        with open(Path(filepath), "w") as f:
            json.dump(data, f)


    def create_fft_animation(self, filepath: Path|str) -> None:
        '''
        Creates an animation file in the provided filepath with the `Normalized Amplitude - Frequency` graph over time
        as produced by the fft.

        Parameters
        ----------
        filepath : `Path | str`
            The filepath where the resulting animation file will be saved (must contain file extension). 
        '''

        if Path(filepath).suffix == ".gif":
            if self.config.framerate > 30:
                self.config.framerate = 30

        self.frame_count = int(self.audio_length_sec * self.config.framerate)
        self.frame_offset = int(len(self.audio)/self.frame_count)

        fft_array = self._create_fft_array()

        start , end = self._get_frequency_ranges(fft_array)
        self.fft_frequency_array = self.fft_frequency_array[start:end]
        # print(fft_array.shape)

        self._create_animation_file(self.fft_frequency_array, fft_array[:, start:end], self._create_fft_frame, filepath)


    def create_bar_animation(self, filepath: Path|str) -> None:
        '''
        Creates an animation file in the provided filepath with the bar graph over time.

        Parameters
        ----------
        filepath : `Path | str`
            The filepath where the resulting animation file will be saved (must contain file extension). 
        '''

        if Path(filepath).suffix == ".gif":
            if self.config.framerate > 30:
                self.config.framerate = 30

        self.frame_count = int(self.audio_length_sec * self.config.framerate)
        self.frame_offset = int(len(self.audio) / self.frame_count)

        fft_array = self._create_fft_array()

        start , end = self._get_frequency_ranges(fft_array)
        if (end - start) < self.config.bars:
            raise ValueError(f'''Not enough frequencies ({end - start}) for bar representation ({self.config.bars}). 
            Please reduce the amount of bars or the amplitude threshold.''')
        
        bar_array: np.ndarray = self._create_bar_array(fft_array, start, end)
        # print(len(bar_array), bar_array.shape)

        x_values = [i for i in range(1, self.config.bars + 1)]
        self._create_animation_file(x_values, bar_array, self._create_bar_graph_frame, filepath)
