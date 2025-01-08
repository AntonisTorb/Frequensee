from __future__ import annotations
import logging
from pathlib import Path

from matplotlib.animation import FFMpegWriter, _log, writers


@writers.register('ffmpeg_custom')
class FFMpegWriterWithAudio(FFMpegWriter):

    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None, metadata=None, audio_filepath: Path|None = None):
        '''
        Custom `FFMpegWriter` class that allows audio integration to the resulting animation video file.

        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the filename. The
            default, None, means to use :rc:`animation.[name-of-encoder]_args` for the
            builtin writers.
        metadata : dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        audio_filepath : Path|None, default: None
            Path to audio file to be added to the animation.
        '''

        self.audio_filepath = audio_filepath
        super().__init__(fps, codec, bitrate, extra_args, metadata)


    def _args(self):

        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie using a pipe.
        args = [self.bin_path(), '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '%dx%d' % self.frame_size, '-pix_fmt', self.frame_format,
                '-framerate', str(self.fps)]
        # Logging is quieted because subprocess.PIPE has limited buffer size.
        # If you have a lot of frames in your animation and set logging to
        # DEBUG, you will have a buffer overrun.
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        if self.audio_filepath is not None:
            args += ['-i', 'pipe:'] + ["-i", f'{self.audio_filepath}'] + self.output_args
        else:
            args += ['-i', 'pipe:'] + self.output_args
        return args