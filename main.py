import numpy as np

def create_file_name(base, suffix):
    ext = ".pgm"
    return base[:-4] + "_" + suffix + ext

def open_ppm_as_array(path_to_file):
    # Open .ppm file
    lines = []
    with open(path_to_file, encoding='latin-1') as f:
        lines = f.readlines()

    # Check header
    assert lines[0] == "P6\n"
    assert len(lines[1].split()) == 2
    assert lines[1].split()[0].isdigit()
    assert lines[1].split()[1].isdigit()
    assert lines[2].split()[0].isdigit()
    assert int(lines[2]) <= 255
    width  = int(lines[1].split()[0])
    height = int(lines[1].split()[1])
    max_value = int(lines[2])

    # Save pixels to array
    assert sum(len(x) for x in lines[3:]) == 3*width*height
    tmp = []
    for x in lines[3:]: # row-wise
        for xx in x: # element-wise
            tmp.append(xx)
    assert len(tmp) == 3*width*height
    rgb_image_array = np.empty([height, 3*width], dtype=np.uint8)
    for j in range(height):
        for i in range(3*width):
            assert ord(tmp[i+j*(3*width)]) <= max_value
            rgb_image_array[j][i] = ord(tmp[i+j*(3*width)])

    return [width, height, max_value, rgb_image_array]

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

def convolution_3x3(im, c, flag):
    assert c.shape == (3, 3)
    h, w = im.shape
    assert h > 2
    assert w > 2

    # border handling switch
    if flag == 0: # crop
        out = np.empty([(h-2), (w-2)], dtype=np.uint8)

        for j in range(h-2):
            for i in range(w-2):
                out[j][i] = int(c[0][0]*im[j][i] + c[0][1]*im[j][i+1] + c[0][2]*im[j][i+2] 
                            + c[1][0]*im[j+1][i] + c[1][1]*im[j+1][i+1] + c[1][2]*im[j+1][i+2]
                            + c[2][0]*im[j+2][i] + c[2][1]*im[j+2][i+1] + c[2][2]*im[j+2][i+2])

    elif flag == 1: # extend
        # TODO?
        out = np.empty([h, w], dtype=np.uint8)

    return out

def convolution(im, c, flag):
    # TODO 5 7 ....
    if c.shape == (3,3):
        return convolution_3x3(im, c, flag)
    else:
        return convolution_3x3(im, c, flag)

def save_pgm_file(image_array, path_to_file):
    height, width = image_array.shape
    with open(path_to_file, 'w') as f:
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
    return scale(fourier_abs_log)

def compute_ESD(x):
    return x * np.conjugate(x)

def compute_energy(x):
    return np.sum(np.power(np.abs(x), 2))

def compute_power(x):
    height, width = x.shape
    return (compute_energy(x)/(height*width))

def compute_SNR(x, y):
    assert x.shape == y.shape
    noise = y - x
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
    for j in range(height):
        for i in range(width):
            out[j][i]=x[j][i]
    return out

### Main ###
images_dir = "images/"
input_image_name = "sabrina.ppm"
#input_image_name = "lena.ppm"
input_image = images_dir + input_image_name

grayscale_image =  create_file_name(input_image, "gray")
dft_image =        create_file_name(input_image, "dft" )
idft_image =       create_file_name(input_image, "idft")
blurred_image =    create_file_name(input_image, "blur")
blurred_image_f =  create_file_name(input_image, "blur_f")
dft_blur_image =   create_file_name(input_image, "dft_blur")
dft_blur_image_f = create_file_name(input_image, "dft_blur_f")
deconv_image =     create_file_name(input_image, "deconv")

# Open .ppm file as array
[width, height, max_value, rgb_image_array] = open_ppm_as_array(input_image)

# Convert RGB array to grayscale
grayscale_image_array = rgb_to_grayscale(rgb_image_array)
save_pgm_file(grayscale_image_array, grayscale_image)

# Fourier
dft_image_array = np.fft.fft2(grayscale_image_array)
fourier_abs_log_scaled = rearrange_FFT(dft_image_array)
save_pgm_file(fourier_abs_log_scaled, dft_image)

# Inverse Fourier
idft_image_array = np.fft.ifft2(dft_image_array)
save_pgm_file((idft_image_array.real).astype(np.uint8), idft_image)

print(compute_SNR_in_dB(grayscale_image_array, (idft_image_array.real).astype(np.uint8) ))

# Gaussian filter
kernel_blur = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
zero_padded_kernel = zero_pad(kernel_blur, dft_image_array.shape[0], dft_image_array.shape[1])
dft_kernel_blur = np.fft.fft2(zero_padded_kernel)
## Convolution in space domain
blurred_image_array = convolution(grayscale_image_array, kernel_blur, 0)
save_pgm_file(blurred_image_array, blurred_image)
save_pgm_file(rearrange_FFT(np.fft.fft2(blurred_image_array)), dft_blur_image)
print(compute_SNR_in_dB(grayscale_image_array[1:-1,1:-1], blurred_image_array ))
# Product in Fourier domain
save_pgm_file(rearrange_FFT(dft_image_array * dft_kernel_blur), dft_blur_image_f)
blurred_image_array = (np.fft.ifft2(dft_image_array * dft_kernel_blur)).real.astype(np.uint8)
save_pgm_file(blurred_image_array, blurred_image_f)
print(compute_SNR_in_dB(grayscale_image_array, blurred_image_array ))

# De-blurring
inverse_dft_kernel_blur = np.empty(dft_kernel_blur.shape, dtype=np.complex128)
snr = 0
th = 0.00001
step = 0.1
par = 1.5
while(step > th): # look for min(SNR); if we don't fall in local min or we too converge rapidly, it should work
    for j in range(512):
        for i in range(512):
            if np.abs(dft_kernel_blur[j][i]) < par: # change this value according to snr
                inverse_dft_kernel_blur[j][i] = 1 # cannot recover
            else:
                inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
    
    deconv_f = np.fft.fft2(blurred_image_array) * inverse_dft_kernel_blur
    deconv_s = np.fft.ifft2(deconv_f)
    snr_new = compute_SNR_in_dB(grayscale_image_array, deconv_s.real.astype(np.uint8) )

    if(snr_new >= snr and par - step > th):
        par -= step
    else:
        step /= 10
        par += step
    snr = snr_new
print(snr)
save_pgm_file(deconv_s.real.astype(np.uint8), deconv_image)

