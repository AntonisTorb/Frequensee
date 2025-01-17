import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import numpy as np


def test_boost(a: float, b: float) -> None:
    image_size_pix = (800, 800)
    dpi = 100
    image_size_inch = tuple([(size_pixel / dpi) for size_pixel in image_size_pix])
    fig, ax = plt.subplots(figsize=image_size_inch, dpi=dpi)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("Input Amplitude", fontsize = 18)
    ax.set_ylabel("Output Amplitude", fontsize = 18)
    
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

    ax.grid(which="both")
    fig.patch.set_color((0.8,0.8,0.8))
    ax.patch.set_alpha(0)
    ax.set_aspect(1)

    x_values = np.linspace(0,1,100)
    y_values = np.log(a*x_values + b) / np.log(a + b)

    ax.plot(x_values, x_values, linewidth=2)
    ax.plot(x_values, y_values, linewidth=2)

    plt.rcParams["mathtext.fontset"] = 'cm'
    formula_enum = f'log({a}X + {b:.2f})'
    formula_denom = f'log({a} + {b:.2f})'
    ax.text(0.4, 0.2, r'$\frac{' + formula_enum + '}{' + formula_denom + r'}$', fontsize=20)

    plt.show()


# Below is the code used to create the `boost.webp` image.

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, FFMpegWriter
# from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

# import numpy as np

# image_size_pix = (800, 800)
# dpi = 100
# image_size_inch = tuple([(size_pixel / dpi) for size_pixel in image_size_pix])
# fig, ax = plt.subplots(figsize=image_size_inch, dpi=dpi)

# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_xlabel("Input Amplitude", fontsize = 18)
# ax.set_ylabel("Output Amplitude", fontsize = 18)
# fig.patch.set_color((0.8,0.8,0.8))
# ax.patch.set_alpha(0)
# ax.set_aspect(1)

# ax.minorticks_on()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
# ax.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
# ax.grid(which="both")

# x_values = np.linspace(0,1,100)
# ax.plot(x_values, x_values, linewidth=2)
# lines = ax.plot(x_values, [0 for _ in x_values], linewidth=2)

# plt.rcParams["mathtext.fontset"] = 'cm'
# text = ax.text(0.4, 0.2, "log(a*X + b) / log(a + b)", fontsize=20)

# a_range = ([i for i in range(1, 101)] + 
#             [i for i in reversed(range(1, 101))] + 
#             [1 for _ in range(202)] +
#             [i for i in range(1, 101)] + 
#             [i for i in reversed(range(1, 101))]
#         )
# b_range = ([1 for _ in range(200)] + 
#             [1 + i/50 for i in range(101)] + 
#             [1 + i/50 for i in reversed(range(101))] + 
#             [1 + i/50 for i in range(1, 101)] + 
#             [1 + i/50 for i in reversed(range(1, 101))]
#         )

# values = [(a,b) for a,b in zip(a_range, b_range)]

# frames = len(a_range)
# fps = 30

# def animate(frame):
#     for line in lines:
#         a , b = values[frame]
#         y = np.log(a*x_values + b) / np.log(a + b)
#         line.set_data(x_values, y)
#     formula_enum = f'log({a}X + {b:.2f})'
#     formula_denom = f'log({a} + {b:.2f})'
#     t = r'$\frac{' + formula_enum + '}{' + formula_denom + r'}$'
#     text.set_text(t)

# ani = FuncAnimation(fig, animate, frames = frames)

# ffmpeg_options = ["-c:v", "webp", "-loop", "0", "-pix_fmt", "yuva420p"]
# writer = FFMpegWriter(fps=fps, extra_args=ffmpeg_options)

# ani.save("boost.webp", writer=writer, dpi=dpi)