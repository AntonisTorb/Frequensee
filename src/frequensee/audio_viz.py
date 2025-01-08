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
from matplotlib.patches import Rectangle
import numpy as np
import soundfile as sf

from .config import Config
from .writers import FFMpegWriterWithAudio


class AudioVisualizer():

    def __init__(self, config: Config) -> None:
        '''
        Audio frequency visualizer using the `Fast Fourier Transform (fft)` in order to produce animations with FFMPEG.

        Parameters
        ----------
        config : Config
            Instance of `Config` class containing configuration options (check `config.py`).
        '''

        self.config = config

        # If FFMPEG is not installed or `ffmpeg.exe` not in working directory, 
        # raise error, unless exporting json.
        if ((shutil.which("ffmpeg") is None and 
            shutil.which(Path.cwd() / "ffmpeg.exe") is None) and 
            not self.config.export_json):
            raise FileNotFoundError("Error: FFMPEG is not installed.")


    def _calculate_audio_fft_properties(self):
        '''Calculates and saves required properties for the fft calculations.'''
        
        self.fft_window_size = int(self.sample_rate * self.config.fft_window_sec)
        self.audio_length_sec: float = self.audio.shape[0] / self.sample_rate
        self.fft_frequency_array: np.ndarray = np.fft.rfftfreq(self.fft_window_size, 1/self.sample_rate)


    def load(self, audio_path: Path|str) -> None:
        '''
        Loads the audio file data and performs calculations for required properties.
        
        Parameters
        ----------
        audio_path : pathlib.Path | str
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
        config : Config
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
        frame_number : int
            Frame number for which to calculate the fft arrray.

        Returns
        -------
        np.ndarray
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
        Creates the fft array and returns it.
        
        Returns
        -------
        np.ndarray
            Numpy array containing the fft data for all frames(rows).
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
        return np.array(fft_array)


    def _get_frequency_ranges(self, fft_array: np.ndarray) -> tuple[int]:
        '''Determines the edge frequencies where the amplitude is high enough to visualize consistently.
        
        Parameters
        ----------
        fft_array : np.ndarray
            Fft array created by the `_create_fft_array` method.

        Returns
        -------
        tuple of int with size of 2
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
                end = len(max_of_cols) -  index
                break

        # print(start, end)
        # print(self.fft_frequency_array[start], self.fft_frequency_array[end])
        return start, end


    def _create_bar_array(self, fft_array: np.ndarray, start: int, end: int) -> np.ndarray:
        '''
        Creates and returns a numpy array containing all the values for the bar visualization.
        The array is split into pieces according to the provided amount of bars and the sum of 
        the values iis used for the bar height. The array is normalized based on the maximum value.

        Parameters
        ----------
        fft_array : np.ndarray
            Fft array created by the `_create_fft_array` method.
        start : int
            Start index for `fft_array`.
        end : int
            End index for `fft_array`.

        Returns
        -------
        np.ndarray
            Numpy array containing the values for the bar visualization.
        '''

        bar_array = []
        for frame in fft_array:
            frame:np.ndarray = frame[start:end]
            chunks: list[np.ndarray] = np.array_split(frame, self.config.bars)
            bar_array.append([chunk.sum() for chunk in chunks])
        bar_array = np.array(bar_array)

        return bar_array / bar_array.max()


    def _create_fft_frame(self, frame_number: int, ax: Axes,
                           x_values: np.ndarray, y_values: np.ndarray) -> list[Artist]:
        '''
        Used by the `_create_animation_file` method. Creates and returns a list of `Artist` objects 
        used to draw the animation of the frequencies over time.

        Parameters
        ----------
        frame_number : int
            The frame number.
        ax : Axes
            Axes object where the graph is drawn.
        x_values : np.ndarray
            Numpy array containing the values for the X axis.
        y_values : np.ndarray
            Numpy array containing the values for the Y axis.
            

        Returns
        -------
        List of `Artist`
            List of Artist used by the animation function.
        '''

        ax.clear()
        ax.set_ylim(0,1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Relative Amplitude")
        ax.patch.set_alpha(0)

        lines = ax.plot(x_values, y_values[frame_number])

        print(f'{(100*frame_number/self.frame_count):.2f}%', end="\r")
        return lines


    def _get_colour_gradient_full(self, h: float) -> np.ndarray:
        '''
        Used by the `_create_bar_graph_frame` method to display a continuous gradient bar image. 
        Creates and returns a numpy array with the RGB values for the image gradient.

        Parameters
        ----------
        h : float
            The height of the bar

        Returns
        -------
        np.ndarray
            Numpy array of RGB values for the gradient.
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
        
        # print(h, r.shape, g.shape, b.shape)
        grad = np.concatenate([r, g, b], axis=2)
        return grad


    def _get_colour_gradient_parts(self, h: float) -> np.ndarray:
        '''
        Used by the `_create_bar_graph_frame` method to display a gradient bar image split in parts.
        Creates and returns a numpy array with the RGBA values for the image gradient.

        Parameters
        ----------
        h : float
            The height of the bar

        Returns
        -------
        np.ndarray
            Numpy array of RGBA values for the gradient.
        '''
        
        # Precalculations to reduce load on every loop.
        bar_parts = self.config.bar_parts
        loop_range = int(100 * self.config.part_gap)
        alpha_lower_limit = loop_range * self.config.part_gap / 2
        alpha_upper_limit = loop_range - alpha_lower_limit

        # `np.linspace`` can't work here since we want non-continuous values. Is there an alternative with np?

        # RGB calculation: the second loop is needed to create a percentage based alpha channel. Need to maintain dimensions.
        r = np.array([
            [
                [self.config.bar_colour_bottom[0] + (self.config.bar_colour_top[0] - self.config.bar_colour_bottom[0]) / bar_parts * i]
            ] for i in reversed(range(int(bar_parts * h))) for _ in range(loop_range)
        ])
        g = np.array([
            [
                [self.config.bar_colour_bottom[1] + (self.config.bar_colour_top[1] - self.config.bar_colour_bottom[1]) / bar_parts * i]
            ] for i in reversed(range(int(bar_parts * h))) for _ in range(loop_range)
        ])
        b = np.array([
            [
                [self.config.bar_colour_bottom[2] + (self.config.bar_colour_top[2] - self.config.bar_colour_bottom[2]) / bar_parts * i]
            ] for i in reversed(range(int(bar_parts * h))) for _ in range(loop_range)
        ])
        # Alpha calculation: 1 only if i between lower and upper limits, else 0.
        a = np.array([
            [
                [int((i > alpha_lower_limit) and (i < alpha_upper_limit))]
            ] for _ in reversed(range(int(bar_parts * h))) for i in range(loop_range)
        ])

        # print(h, r.shape, g.shape, b.shape, a.shape)
        grad = np.concatenate([r, g, b, a], axis=2)
        return grad


    def _create_bar_graph_frame(self, frame_number:int, ax: Axes, 
                                x_values: list[int], y_values: np.ndarray) -> list[Artist]:
        '''
        Used by the `_create_animation_file` method. Creates and returns a list of `Artist` objects 
        used to draw the animation of the bars over time.

        Parameters
        ----------
        frame_number : int
            The frame number.
        ax : Axes
            Axes object where the graph is drawn.
        x_values : np.ndarray
            Numpy array containing the values for the X axis.
        y_values : np.ndarray
            Numpy array containing the values for the Y axis.
            

        Returns
        -------
        List of `Artist`
            List of Artist used by the animation function.
        '''

        ax.clear()
        plt.axis("off")
        plt.margins(x=0)
        ax.set_ylim(0,1)
        ax.patch.set_alpha(0)
        lim = ax.get_xlim() + ax.get_ylim()
        ax.axis(lim)
        
        bar: Rectangle
        bars = ax.bar(x_values, y_values[frame_number])

        images: list[AxesImage] = []

        # Replace bars with color gradient image based on bar height.
        for bar in bars:
            bar.set_alpha(0)

            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()

            if h == 0:
                continue

            if self.config.bar_parts:
                if h <= 1 / self.config.bar_parts:
                    continue

                # Normalize the height so bar parts align.
                if h != 0:
                    h = ceil(self.config.bar_parts * h) / self.config.bar_parts

            if self.config.bar_parts:
                grad = self._get_colour_gradient_parts(h)
            else:
                grad = self._get_colour_gradient_full(h)
            
            img = ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=10)
            images.append(img)

        print(f'{(100*frame_number/self.frame_count):.2f}%', end="\r")
        return images


    def _create_writer(self, filepath) -> FFMpegWriter|FFMpegWriterWithAudio:
        '''
        Creates and returns the writer object for the `_create_animation_file` method in order to save the animation file.

        Parameters
        ----------
        filepath : Path|str
            The filepath to the output file.

        Returns
        -------
        FFMpegWriter or FFMpegWriterWithAudio
            Appropriate writer initialized with options according to output file format.
        '''

        if Path(filepath).suffix in [".gif", ".webp"]:
            if Path(filepath).suffix == ".webp":
                if self.config.ffmpeg_options is not None:
                    self.config.ffmpeg_options = ["-c:v", "webp", "-loop", "0", "-pix_fmt", "yuva420p"] + self.config.ffmpeg_options
                else:
                    self.config.ffmpeg_options = ["-c:v", "webp", "-loop", "0", "-pix_fmt", "yuva420p"]

            writer = FFMpegWriter(fps=self.config.framerate, extra_args=self.config.ffmpeg_options)
        else:
            writer = FFMpegWriterWithAudio(fps=self.config.framerate, audio_filepath=self.audio_path, extra_args=self.config.ffmpeg_options)

        return writer


    def _create_animation_file(self, x_values: list|np.ndarray, y_values: np.ndarray, 
                               animation_func: Callable, filepath: Path|str) -> None:
        '''
        Creates the animation file(s) in the specified path.

        Parameters
        ----------
        x_values : np.ndarray
            Numpy array containing the values for the X axis.
        y_values : np.ndarray
            Numpy array containing the values for the Y axis.
        animation_func: Callable
            Function to be used by `FuncAnimation` to produce the animation frames.
        filepath: Path|str
            The filepath where the resulting animation file(s) will be saved (must contain file extension).
        '''

        self.image_size_inch = tuple([(size_pixel / self.config.dpi) for size_pixel in self.config.image_size_pix])
        fig, ax = plt.subplots(figsize=self.image_size_inch)
        fig.patch.set_alpha(self.config.background[-1])
        fig.patch.set_color(self.config.background[:-1])
        if animation_func == self._create_bar_graph_frame:
            fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0) 

        writer = self._create_writer(filepath)

        if (Path(filepath).suffix == ".gif" and 
            self.config.max_frames_per_gif and 
            y_values.shape[0] > self.config.max_frames_per_gif):

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
                            partial(animation_func, ax=ax, x_values=x_values, y_values=y_values), 
                            frames = y_values.shape[0])

                ani.save(filepath_part, writer=writer, dpi=self.config.dpi,
                 savefig_kwargs={"transparent": True})
            print(f'Done.{" "*20}')
            plt.close()
            return
        
        print(f'Creating {filepath}...')
        ani = FuncAnimation(fig, 
                            partial(animation_func, ax=ax, x_values=x_values, y_values=y_values), 
                            frames = y_values.shape[0])

        ani.save(filepath, writer=writer, dpi=self.config.dpi,
                 savefig_kwargs={"transparent": True})

        print(f'Done.{" "*20}')
        plt.close()
        

    def extract_json(self, filepath: Path|str) -> None:
        '''
        Extracts the audio filename, ammount of bars and the bar array in a JSON file.
        
        Parameters
        ----------
        filepath : Path | str
            The filepath where the resulting `json` file will be saved (must contain `.json` file extension). 
        '''

        self.frame_count = int(self.audio_length_sec * self.config.framerate)
        self.frame_offset = int(len(self.audio)/self.frame_count)

        fft_array = self._create_fft_array()
        fft_array /= fft_array.max()
        start, end = self._get_frequency_ranges(fft_array)
        bar_array: np.ndarray = self._create_bar_array(fft_array, start, end)

        data = {
            "audio_filepath": self.audio_path,
            "bars": self.config.bars,
            "bar_graph": bar_array.tolist()
        }

        with open(Path(filepath), "w") as f:
            json.dump(data, f)


    def create_fft_animation(self, filepath: Path|str) -> None:
        '''
        Creates an animation file in the provided filepath with the Amplitude - Frequency graph over time
        as produced with the fft.

        Parameters
        ----------
        filepath : Path | str
            The filepath where the resulting animation file will be saved (must contain file extension). 
        '''

        # add user defined start and end frequencies for visualization.

        if Path(filepath).suffix == ".gif":
            if self.config.framerate > 30:
                self.config.framerate = 30

        self.frame_count = int(self.audio_length_sec * self.config.framerate)
        self.frame_offset = int(len(self.audio)/self.frame_count)

        fft_array = self._create_fft_array()
        # Normalize between 0 and 1.
        fft_array: np.ndarray = fft_array / fft_array.max()

        start , end = self._get_frequency_ranges(fft_array)
        # print(fft_array.shape)

        self._create_animation_file(self.fft_frequency_array[start:end], fft_array[:, start:end], self._create_fft_frame, filepath)


    def create_bar_animation(self, filepath: Path|str):
        '''
        Creates an animation file in the provided filepath with the bar graph over time.

        Parameters
        ----------
        filepath : Path | str
            The filepath where the resulting animation file will be saved (must contain file extension). 
        '''

        if Path(filepath).suffix == ".gif":
            if self.config.framerate > 30:
                self.config.framerate = 30

        self.frame_count = int(self.audio_length_sec * self.config.framerate)
        self.frame_offset = int(len(self.audio) / self.frame_count)

        fft_array = self._create_fft_array()
        # Normalize between 0 and 1.
        fft_array: np.ndarray = fft_array / fft_array.max()

        start , end = self._get_frequency_ranges(fft_array)
        if (end - start) < self.config.bars:
            raise ValueError(f'''Not enough frequencies ({end - start}) for bar representation ({self.config.bars}). 
            Please reduce the amount of bars or the amplitude threshold.''')
        
        bar_array: np.ndarray = self._create_bar_array(fft_array, start, end)
        # print(len(bar_array), bar_array.shape)

        x_values = [i for i in range(1, self.config.bars + 1)]
        self._create_animation_file(x_values, bar_array, self._create_bar_graph_frame, filepath)
