from functions import *
import sys

images_dir = "images/"
input_images_dir = images_dir + "sample_input/"
output_images_dir = images_dir + "results/"

if(len(sys.argv) > 1):
  if(sys.argv[1] == 'sabrina'):
    input_image_name = "sabrina.ppm"
  elif(sys.argv[1] == 'lena'):
    input_image_name = "lena.ppm"
  else:
    raise Exception("Usage: main.py <sabrina or lena>") 
else:
  # Default value
  input_image_name = "sabrina.ppm"
  #input_image_name = "lena.ppm"

# Ouput file names
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
height, width = rgb_image_array.shape
width //= 3
print("Height: " + str(height) + "; Width: " + str(width))

# Convert RGB array to grayscale
grayscale_image_array = rgb_to_grayscale(rgb_image_array)
save_pgm_file(grayscale_image_array, grayscale_image)
print("> Created grayscale " + grayscale_image)

# Symmetrization to avoid introduciton of high freqs
symm_image_array = symmetrization(grayscale_image_array)
#save_pgm_file(symm_image_array, symmetric_image)
#print("> Created symmetric grayscale " + symmetric_image)

# FFT - IFFT
#dft_image_array = np.fft.fft2(grayscale_image_array)
#idft_image_array = np.fft.ifft2(dft_image_array)
#print("SNR after FFT and IFFT: " + str(compute_SNR_in_dB(grayscale_image_array, idft_image_array.real.astype(np.uint8))) + " dB")

# FFT - IIF with symmetrization
dft_image_array = np.fft.fft2(symm_image_array)
idft_image_array = np.fft.ifft2(dft_image_array)
#save_pgm_file(idft_image_array[0:width, 0:height].real.astype(np.uint8), idft_image)
#print("> Created IDFT " + idft_image)
fourier_abs_log_scaled = rearrange_FFT(dft_image_array)
save_pgm_file(fourier_abs_log_scaled, dft_image)
print("> Created abs log scaled DFT " + dft_image)
#print("SNR after FFT and IFFT (with symmetrization): " + str(compute_SNR_in_dB(grayscale_image_array, idft_image_array.real[0:width, 0:height].astype(np.uint8))) + " dB")

# Convolution in space domain
kernel_blur = gaussian_kernel(3)
blurred_image_array = convolution(grayscale_image_array, kernel_blur, 1)
save_pgm_file(blurred_image_array, blurred_image)
print("> Created blurred " + symmetric_image)
dft_blurred_image = np.fft.fft2(symmetrization(blurred_image_array))
save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image)
print("> Created abs log scaled DFT blurred " + dft_blur_image)
print("SNR after blurring: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])) + " dB")

## Product in Fourier domain
#dft_blurred_image = dft_image_array * dft_kernel_blur
#save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image_f)
#print("> Created DFT blurred in Fourier " + dft_blur_image_f)
#blurred_image_array = (np.fft.ifft2(dft_image_array * dft_kernel_blur)).real.astype(np.uint8)
#save_pgm_file(blurred_image_array[0:width, 0:height], blurred_image_f)
#print("> Created blurred in Fourier " + blurred_image_f)
#print("SNR after blurring in Fourier: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])) + " dB")

# Inverse kernel
zero_padded_kernel = zero_pad(kernel_blur, symm_image_array.shape[0], symm_image_array.shape[1])
dft_kernel_blur = np.fft.fft2(zero_padded_kernel)
save_pgm_file(rearrange_FFT(dft_kernel_blur), dft_kernel_image)
print("> Created abs log scaled DFT kernel " + dft_kernel_image)

#limits = np.arange(0.0, 1.0, 0.01)
limits = [0.07]
snrs = []
for limit in limits:
  inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)
  for j in range(inverse_dft_kernel_blur.shape[0]):
      for i in range(inverse_dft_kernel_blur.shape[1]):
          #print(np.abs(dft_kernel_blur[j][i]))
          if np.abs(dft_kernel_blur[j][i]) <= limit:
              inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
          else:
              inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
          #print(np.abs(inverse_dft_kernel_blur[j][i]))
  #save_pgm_file(rearrange_FFT(inverse_dft_kernel_blur), inverse_kernel_image)
  
  deconv_f = dft_blurred_image * inverse_dft_kernel_blur
  save_pgm_file(rearrange_FFT(deconv_f), dft_deconv_image)
  print("> Created abs log scaled DFT de-blurred " + dft_deconv_image)
  deconv_s = np.fft.ifft2(deconv_f)
  snr = compute_SNR_in_dB(grayscale_image_array, deconv_s[width:, height:].real.astype(np.uint8))
  print("SNR after deblurring: " + str(snr) + " dB")
  snrs.append(snr)
  save_pgm_file(deconv_s[width:, height:].real.astype(np.uint8), deconv_image)
  print("> Created de-blurred image " + deconv_image)

# TODO Non sicurissimo di come fare lo zero padding e di come ritagliare l'immagine finale..., ci sono delle rotazioni di mezzo

