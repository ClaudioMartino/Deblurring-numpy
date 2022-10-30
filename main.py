import numpy as np

def create_file_name(directory, base, suffix):
    ext = ".pgm"
    return directory + "/" + base[:-4] + "_" + suffix + ext

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



### Main ###
images_dir = "images/"
input_images_dir = images_dir + "sample_input/"
output_images_dir = images_dir + "results/"

#input_image_name = "sabrina.ppm"
input_image_name = "lena.ppm"

grayscale_image =  create_file_name(output_images_dir, input_image_name, "gray")
symmetric_image =  create_file_name(output_images_dir, input_image_name, "symm")
dft_image =        create_file_name(output_images_dir, input_image_name, "dft" )
idft_image =       create_file_name(output_images_dir, input_image_name, "idft")
blurred_image =    create_file_name(output_images_dir, input_image_name, "blur")
blurred_image_f =  create_file_name(output_images_dir, input_image_name, "blur_f")
dft_blur_image =   create_file_name(output_images_dir, input_image_name, "dft_blur")
dft_blur_image_f = create_file_name(output_images_dir, input_image_name, "dft_blur_f")
deconv_image =     create_file_name(output_images_dir, input_image_name, "deconv")
dft_deconv_image = create_file_name(output_images_dir, input_image_name, "dft_deconv")
inverse_kernel_image = create_file_name(output_images_dir, input_image_name, "inv_kernel")
dft_kernel_image = create_file_name(output_images_dir, input_image_name, "dft_kernel")

# Open .ppm file as array
input_image = input_images_dir + input_image_name
rgb_image_array = open_ppm_as_array(input_image)

# Convert RGB array to grayscale
grayscale_image_array = rgb_to_grayscale(rgb_image_array)
height, width = grayscale_image_array.shape
save_pgm_file(grayscale_image_array, grayscale_image)

# FFT - IFFT
dft_image_array = np.fft.fft2(grayscale_image_array)
idft_image_array = np.fft.ifft2(dft_image_array)

print("SNR FFT/IFFT: " + str(compute_SNR_in_dB(grayscale_image_array, idft_image_array.real.astype(np.uint8)) ))

# FFT - IIF with symmetrization
symm_image_array = symmetrization(grayscale_image_array)
save_pgm_file(symm_image_array, symmetric_image)

dft_image_array = np.fft.fft2(symm_image_array)
fourier_abs_log_scaled = rearrange_FFT(dft_image_array)
save_pgm_file(fourier_abs_log_scaled, dft_image)

idft_image_array = np.fft.ifft2(dft_image_array)
save_pgm_file(idft_image_array[0:width, 0:height].real.astype(np.uint8), idft_image)

print("SNR FFT/IFFT (symm.): " + str(compute_SNR_in_dB(grayscale_image_array, idft_image_array.real[0:width, 0:height].astype(np.uint8)) ))

# Gaussian filter
kernel_blur = gaussian_kernel(7)
zero_padded_kernel = zero_pad(kernel_blur, dft_image_array.shape[0], dft_image_array.shape[1])
dft_kernel_blur = np.fft.fft2(zero_padded_kernel)
## Convolution in space domain
blurred_image_array = convolution(grayscale_image_array, kernel_blur, 1)
save_pgm_file(blurred_image_array, blurred_image)
dft_blurred_image = np.fft.fft2(symmetrization(blurred_image_array))
print(np.max(np.imag(dft_blurred_image)))
save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image)
print("SNR conv: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height]) ))
## Product in Fourier domain
#dft_blurred_image = dft_image_array * dft_kernel_blur
#print(np.max(np.imag(dft_blurred_image)))
#save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image_f)
#blurred_image_array = (np.fft.ifft2(dft_image_array * dft_kernel_blur)).real.astype(np.uint8)
#save_pgm_file(blurred_image_array[0:width, 0:height], blurred_image_f)
#print("SNR prod: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height]) ))

# De-blurring
inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)
limit = 0

#l=10
#max_lim = np.max(np.abs(dft_kernel_blur))
#min_lim = np.min(np.abs(dft_kernel_blur))
#snr = np.zeros(l)
#diff_snr_th = 1e-1
#diff_snr = 999
#cnt=0
#while(diff_snr > diff_snr_th and cnt<5):
#    limit_array = np.linspace(max_lim, min_lim, l)
#    step = np.abs(max_lim-min_lim)/(l-1)
#    k=0
#    for limit in limit_array:
#        for j in range(inverse_dft_kernel_blur.shape[0]):
#            for i in range(inverse_dft_kernel_blur.shape[1]):
#                if np.abs(dft_kernel_blur[j][i]) <= limit: # change limit value according to SNR
#                    inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
#                else:
#                    inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
#    
#        deconv_f = dft_blurred_image * inverse_dft_kernel_blur
#        deconv_s = np.fft.ifft2(deconv_f)
#        snr[k] = compute_SNR_in_dB(grayscale_image_array, deconv_s[0:width, 0:height].real.astype(np.uint8) )
#        print("limit: " + str(limit) + " SNR deblur: " + str(snr[k]))
#        k += 1
#    ind_max = np.argmax(snr)
#    ind_max_prev = max(0, ind_max-1)
#    ind_max_next = min(ind_max+1, l-1)
#    max_lim = limit_array[ind_max_next]
#    min_lim = limit_array[ind_max_prev]
#    diff_snr1 = np.abs(snr[ind_max] - snr[ind_max_prev])
#    diff_snr2 = np.abs(snr[ind_max] - snr[ind_max_next])
#    diff_snr = max(diff_snr1, diff_snr2)
#    print(diff_snr)
#    cnt += 1
#limit = limit_array[ind_max]

print(inverse_dft_kernel_blur.shape)
print(dft_kernel_blur.shape)
for j in range(inverse_dft_kernel_blur.shape[0]):
    for i in range(inverse_dft_kernel_blur.shape[1]):
        if np.abs(dft_kernel_blur[j][i]) <= limit: # change limit value according to SNR
            inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
        else:
            inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
        #print(np.abs(inverse_dft_kernel_blur[j][i]))

deconv_f = dft_blurred_image * inverse_dft_kernel_blur
save_pgm_file(rearrange_FFT(deconv_f), dft_deconv_image)
deconv_s = np.fft.ifft2(deconv_f)
print("SNR deblur: " + str(compute_SNR_in_dB(grayscale_image_array, deconv_s[0:width, 0:height].real.astype(np.uint8))))
save_pgm_file(deconv_s[0:width, 0:height].real.astype(np.uint8), deconv_image)

save_pgm_file(rearrange_FFT(dft_kernel_blur), dft_kernel_image)
save_pgm_file(rearrange_FFT(inverse_dft_kernel_blur), inverse_kernel_image)

# i valori di lena dft deconv son diversi
