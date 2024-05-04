import numpy as np

def create_file_name(directory, base, suffix):
    ext = ".pgm"
    return directory + "/" + base[:-4] + "_" + suffix + ext

def open_ppm_as_array(path_to_file):
    lines = []
    # Open .ppm file
    with open(path_to_file, "rb") as f:
        # Check header
        line = f.readline()
        assert line == b'P6\n'

        line = f.readline().split()
        width  = int(line[0])
        height = int(line[1])

        line = f.readline()
        max_value = int(line[:-1])
        assert max_value <= 255

        # Read pixels
        lines = f.read()
        assert len(lines) == 3*width*height

    # Save pixels to 2D array
    rgb_image_array = np.empty([height, 3*width], dtype=np.uint8)
    for j in range(height):
        for i in range(3*width):
            rgb_image_array[j][i] = lines[i+j*(3*width)]

    return rgb_image_array

def rgb_to_grayscale(rgb_image):
    height, width = rgb_image.shape
    assert width%3 == 0

    # Convert RGB to grayscale
    grayscale_image_array = np.empty([height, int(width/3)], dtype=np.uint8)
    for j in range(height):
        for i in range(int(width/3)):
            red   = rgb_image[j][3*i]
            green = rgb_image[j][3*i+1]
            blue  = rgb_image[j][3*i+2]
            grayscale_image_array[j][i] = int(0.2989*red + 0.5870*green + 0.1140*blue)

    return grayscale_image_array

def convolution(im, c, flag):
    assert c.shape[0] == c.shape[1]
    n = c.shape[0]

    assert n in (3,5,7)
    brd = int((n-1)/2)

    h, w = im.shape
    assert h-n+1 > 0
    assert w-n+1 > 0

    if flag == 0: # crop borders
        # TODO never tested
        out = np.zeros([(h-n+1), (w-n+1)], dtype=np.uint8)

        for j in range(h-n+1):
            for i in range(w-n+1):
                tmp = 0
                for jj in range(n):
                    for ii in range(n):
                        tmp += c[jj][ii]*im[j+jj][i+ii]
                out[j][i] = int(tmp)

    elif flag == 1: # extend borders
        out = np.zeros([h, w], dtype=np.uint8)

        for j in range(h-n+1):
            for i in range(w-n+1):
                tmp = 0
                for jj in range(n):
                    for ii in range(n):
                        tmp += c[jj][ii]*im[j+jj][i+ii]
                out[j+brd][i+brd] = int(tmp)

        # upper border
        for j in range(brd):
            for i in range(w-n+1):
                out[j][i+brd] = out[brd][i+brd]

        # lower border
        for j in range(brd):
            for i in range(w-n+1):
                out[j+brd+h-n+1][i+brd] = out[h-brd-1][i+brd]

        # left border + corners
        for j in range(h):
            for i in range(brd):
                out[j][i] = out[j][brd]

        # right border + corners
        for j in range(h):
            for i in range(brd):
                out[j][i+h-n+1+brd] = out[j][w-brd-1]

    return out


def save_pgm_file(image_array, path_to_file):
    height, width = image_array.shape
    with open(path_to_file, 'w+') as f:
        # Write header
        f.write("P2\n")
        f.write(str(width) + " " + str(height) + "\n")
        f.write("255\n")
        # Write pixels
        for j in range(height):
            for i in range(width):
                f.write(str(image_array[j][i]) + " ")

def scale(x):
    return (x * 255.0/np.max(x)).astype(np.uint8)

def rearrange_FFT(fourier):
    # Shift the zero frequency to the center
    fourier_shifted = np.fft.fftshift(fourier)
    # Take the absolute value
    fourier_abs = np.abs(fourier_shifted)
    # Take the logarithm of the absolute value + 1
    fourier_abs_log = np.log10(fourier_abs + 1)
    # Scale between 0 and 255
    return scale(fourier_abs_log).astype(np.uint8)

def compute_ESD(x):
    return x * np.conjugate(x)

def compute_energy(x):
    return np.sum(np.power(np.abs(x), 2))

def compute_power(x):
    height, width = x.shape
    return (compute_energy(x)/(height*width))

def compute_SNR(x, y):
    assert x.shape == y.shape
    noise = (y.astype(np.int16) - x.astype(np.int16))
    noise_power = compute_power(noise)
    signal_power = compute_power(x)
    return signal_power/noise_power

def compute_SNR_in_dB(x, y):
    return 10*np.log10(compute_SNR(x, y))

def zero_pad(x, height_new, width_new):
    height, width = x.shape
    assert width_new >= width 
    assert height_new >= height
    out = np.zeros([height_new, width_new], dtype=(x.dtype))
    # The kernel is shifted to occupy the 4 corner.
    # The middle pixel is in the upper-left corner.
    for j in range(height):
      for i in range(width):
        out[0 + j - height//2][0 + i - width//2] = x[0 + j][0 + i]

    return out

def gaussian_kernel(size):
    assert size in [3,5,7]

    if size == 3:
        return np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16
    elif size == 5:
        return np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273
    elif size == 7:
        return np.array([[0,0,1,2,1,0,0],[0,3,13,22,13,3,0],[1,13,59,97,59,13,1],[2,22,97,159,97,22,2],[1,13,59,97,59,13,1],[0,3,13,22,13,3,0],[0,0,1,2,1,0,0]]) / 1003

def symmetrization(im):
    height, width = im.shape

    out = np.zeros([height*2, width*2], dtype=(im.dtype))

    for j in range(height):
        for i in range(width):
            out[j][i] = im[j][i]
            out[j][i+width] = im[j][width-1-i]
            out[j+height][i] = im[height-1-j][i]
            out[j+height][i+width] = im[height-1-j][width-1-i]

    return out


