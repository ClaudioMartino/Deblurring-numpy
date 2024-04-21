from functions import *

images_dir = "images/"
input_images_dir = images_dir + "sample_input/"
output_images_dir = images_dir + "results/"

input_image_name = "sabrina.ppm"
#input_image_name = "lena.ppm"

grayscale_image      = create_file_name(output_images_dir, input_image_name, "gray")
symmetric_image      = create_file_name(output_images_dir, input_image_name, "symm")
dft_image            = create_file_name(output_images_dir, input_image_name, "dft" )
idft_image           = create_file_name(output_images_dir, input_image_name, "idft")
blurred_image        = create_file_name(output_images_dir, input_image_name, "blur")
blurred_image_f      = create_file_name(output_images_dir, input_image_name, "blur_f")
dft_blur_image       = create_file_name(output_images_dir, input_image_name, "dft_blur")
dft_blur_image_f     = create_file_name(output_images_dir, input_image_name, "dft_blur_f")
deconv_image         = create_file_name(output_images_dir, input_image_name, "deconv")
dft_deconv_image     = create_file_name(output_images_dir, input_image_name, "dft_deconv")
inverse_kernel_image = create_file_name(output_images_dir, input_image_name, "inv_kernel")
dft_kernel_image     = create_file_name(output_images_dir, input_image_name, "dft_kernel")

# Open .ppm file as array
print("Opening " + input_image_name)
input_image = input_images_dir + input_image_name
rgb_image_array = open_ppm_as_array(input_image)

# Convert RGB array to grayscale
grayscale_image_array = rgb_to_grayscale(rgb_image_array)
height, width = grayscale_image_array.shape
save_pgm_file(grayscale_image_array, grayscale_image)
print("> Created grayscale " + grayscale_image)

# FFT - IFFT
dft_image_array = np.fft.fft2(grayscale_image_array)
idft_image_array = np.fft.ifft2(dft_image_array)
print("SNR after FFT and IFFT: " + str(compute_SNR_in_dB(grayscale_image_array, idft_image_array.real.astype(np.uint8))) + " dB")

# FFT - IIF with symmetrization
symm_image_array = symmetrization(grayscale_image_array)
save_pgm_file(symm_image_array, symmetric_image)
print("> Created symmetric grayscale " + symmetric_image)

dft_image_array = np.fft.fft2(symm_image_array)
fourier_abs_log_scaled = rearrange_FFT(dft_image_array)
save_pgm_file(fourier_abs_log_scaled, dft_image)
print("> Created abs log scaled DFT " + dft_image)

idft_image_array = np.fft.ifft2(dft_image_array)
save_pgm_file(idft_image_array[0:width, 0:height].real.astype(np.uint8), idft_image)
print("> Created IDFT " + idft_image)

print("SNR after FFT and IFFT (with symmetrization): " + str(compute_SNR_in_dB(grayscale_image_array, idft_image_array.real[0:width, 0:height].astype(np.uint8))) + " dB")

# Gaussian filter
kernel_blur = gaussian_kernel(7)
zero_padded_kernel = zero_pad(kernel_blur, dft_image_array.shape[0], dft_image_array.shape[1])
dft_kernel_blur = np.fft.fft2(zero_padded_kernel)

## Convolution in space domain
blurred_image_array = convolution(grayscale_image_array, kernel_blur, 1)
save_pgm_file(blurred_image_array, blurred_image)
print("> Created blurred " + symmetric_image)
dft_blurred_image = np.fft.fft2(symmetrization(blurred_image_array))
save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image)
print("> Created DFT blurred " + dft_blur_image)
print("SNR after blurring: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])) + " dB")

## Product in Fourier domain TODO This works better than convolution, but why?
#dft_blurred_image = dft_image_array * dft_kernel_blur
#save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image_f)
#print("> Created DFT blurred in Fourier " + dft_blur_image_f)
#blurred_image_array = (np.fft.ifft2(dft_image_array * dft_kernel_blur)).real.astype(np.uint8)
#save_pgm_file(blurred_image_array[0:width, 0:height], blurred_image_f)
#print("> Created blurred in Fourier " + blurred_image_f)
#print("SNR after blurring in Fourier: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])) + " dB")

# De-blurring
inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)

limit = 0.01 # TODO
## Use limit to compute kernel
for j in range(inverse_dft_kernel_blur.shape[0]):
    for i in range(inverse_dft_kernel_blur.shape[1]):
        #print(np.abs(dft_kernel_blur[j][i]))
        if np.abs(dft_kernel_blur[j][i]) <= limit: # change limit value according to SNR
            inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
        else:
            inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
        #print(np.abs(inverse_dft_kernel_blur[j][i]))

deconv_f = dft_blurred_image * inverse_dft_kernel_blur
save_pgm_file(rearrange_FFT(deconv_f), dft_deconv_image)
print("> Created DFT de-blurred " + dft_deconv_image)
deconv_s = np.fft.ifft2(deconv_f)
print("SNR after deblurring: " + str(compute_SNR_in_dB(grayscale_image_array, deconv_s[0:width, 0:height].real.astype(np.uint8))))
save_pgm_file(deconv_s[0:width, 0:height].real.astype(np.uint8), deconv_image)
print("> Created de-blurred image " + deconv_image)

save_pgm_file(rearrange_FFT(dft_kernel_blur), dft_kernel_image)
save_pgm_file(rearrange_FFT(inverse_dft_kernel_blur), inverse_kernel_image)

