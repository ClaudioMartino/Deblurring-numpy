from functions import *
from plot import *
import argparse

valid_inputs = ['sabrina', 'lena', 'sara', 'laura', 'all']
valid_kernel_sizes = [3, 5, 7, 'all']

parser = argparse.ArgumentParser(description="Deblurring", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-image", choices=valid_inputs, default=[valid_inputs[1]], nargs="+", help="Input image")
parser.add_argument("-k", "--kernel-size", type=int, choices=valid_kernel_sizes, default=[valid_kernel_sizes[0]], nargs="+", help="Kernel size")
parser.add_argument("-t", "--threshold", type=float, default=0.07, help="Inversion threshold")
parser.add_argument("-p", "--plot", action='store_true', help="Run the SNRs for different thresholds and plot the result")

args = parser.parse_args()
config = vars(args)
print(config)

# Directories
images_dir = "images/"
input_images_dir = images_dir + "sample_input/"
output_images_dir = images_dir + "results/"

# Read arguments
input_image_name_arr = config['input_image']
if(input_image_name_arr == ['all']):
  input_image_name_arr = valid_inputs[:-1]
input_image_name_arr = [x + '.ppm' for x in input_image_name_arr]

kernel_sizes = config['kernel_size']
if(kernel_sizes == ['all']):
  kernel_sizes = valid_kernel_sizes[:-1]

threshold = config['threshold']

plot = config['plot']

# Loop over each image
snrs_kernel_image = []
for input_image_name in input_image_name_arr:
  # Ouput file names
  grayscale_image      = create_file_name(output_images_dir, input_image_name, "gray")
  symmetric_image      = create_file_name(output_images_dir, input_image_name, "symm")
  dft_image            = create_file_name(output_images_dir, input_image_name, "dft" )
  idft_image           = create_file_name(output_images_dir, input_image_name, "idft")

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

  # Loop over each kernel size
  snrs_kernel = []
  for kernel_size in kernel_sizes:
    blurred_image        = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "blur")
    blurred_image_f      = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "blur_f")
    dft_blur_image       = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_blur")
    dft_blur_image_f     = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_blur_f")
    deconv_image         = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "deconv")
    dft_deconv_image     = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_deconv")
    # TODO Questi son troppi
    inverse_kernel_image = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "inv_kernel")
    dft_kernel_image     = create_file_name(output_images_dir, str(kernel_size) + '_' + input_image_name, "dft_kernel")
   
    # 2D FFT of input image
    dft_image_array = np.fft.fft2(symm_image_array)
    fourier_abs_log_scaled = rearrange_FFT(dft_image_array)
    save_pgm_file(fourier_abs_log_scaled, dft_image)
    print("> Created abs log scaled DFT " + dft_image)

    # Inverse 2D FFT of input image
    idft_image_array = np.fft.ifft2(dft_image_array)
    #save_pgm_file(idft_image_array[0:width, 0:height].real.astype(np.uint8), idft_image)
    #print("> Created IDFT " + idft_image)
   
    # Convolution in space domain
    kernel_blur = gaussian_kernel(kernel_size)
    blurred_image_array = convolution(grayscale_image_array, kernel_blur, 1)
    save_pgm_file(blurred_image_array, blurred_image)
    print("> Created blurred " + symmetric_image)
    dft_blurred_image = np.fft.fft2(symmetrization(blurred_image_array))
    save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image)
    print("> Created abs log scaled DFT blurred " + dft_blur_image)
    blur_snr = compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])
    print("SNR after blurring: " + str(blur_snr) + " dB")
    
    # Inverse kernel
    # TODO zero pad con gaussiana divisa in 4 ai 4 angoli, secondo me si evitano rotazioni
    zero_padded_kernel = zero_pad(kernel_blur, symm_image_array.shape[0], symm_image_array.shape[1])
    dft_kernel_blur = np.fft.fft2(zero_padded_kernel)
    save_pgm_file(rearrange_FFT(dft_kernel_blur), dft_kernel_image)
    print("> Created abs log scaled DFT kernel " + dft_kernel_image)
    
    # Product in Fourier domain
    #dft_blurred_image = dft_image_array * dft_kernel_blur
    #save_pgm_file(rearrange_FFT(dft_blurred_image), dft_blur_image_f)
    #print("> Created DFT blurred in Fourier " + dft_blur_image_f)
    #blurred_image_array = (np.fft.ifft2(dft_image_array * dft_kernel_blur)).real.astype(np.uint8)
    #save_pgm_file(blurred_image_array[0:width, 0:height], blurred_image_f)
    #print("> Created blurred in Fourier " + blurred_image_f)
    #print("SNR after blurring in Fourier: " + str(compute_SNR_in_dB(grayscale_image_array, blurred_image_array[0:width, 0:height])) + " dB")
    
    plot_step = 0.01
    limits = np.arange(0.0+plot_step, 1.0, plot_step)
    snrs = []
    if(plot):
      for limit in limits:
        print("Iteration " + str(limit))
        inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)
        for j in range(inverse_dft_kernel_blur.shape[0]):
            for i in range(inverse_dft_kernel_blur.shape[1]):
                if np.abs(dft_kernel_blur[j][i]) <= limit:
                    inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
                else:
                    inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
        
        deconv_f = dft_blurred_image * inverse_dft_kernel_blur
        deconv_s = np.fft.ifft2(deconv_f)
        deconv_s_crop = deconv_s[width:, height:].real
        # TODO separate function Clip
        for y in range(height):
          for x in range(width):
            if(deconv_s_crop[y][x] < 0):
              deconv_s_crop[y][x] = 0
            if(deconv_s_crop[y][x] > 255):
              deconv_s_crop[y][x] = 255
        snr = compute_SNR_in_dB(grayscale_image_array, deconv_s_crop.astype(np.uint8))
        snrs.append(snr - blur_snr)
      # For each kernel size, append the SNRs
      snrs_kernel.append(snrs)
    
    inverse_dft_kernel_blur = np.zeros(dft_kernel_blur.shape, dtype=np.complex128)
    for j in range(inverse_dft_kernel_blur.shape[0]):
        for i in range(inverse_dft_kernel_blur.shape[1]):
            if np.abs(dft_kernel_blur[j][i]) <= threshold:
                inverse_dft_kernel_blur[j][i] = 1 # smaller than limit, cannot recover
            else:
                inverse_dft_kernel_blur[j][i] = 1/dft_kernel_blur[j][i]
            #print(np.abs(inverse_dft_kernel_blur[j][i]))
    #save_pgm_file(rearrange_FFT(inverse_dft_kernel_blur), inverse_kernel_image)
    
    deconv_f = dft_blurred_image * inverse_dft_kernel_blur
    save_pgm_file(rearrange_FFT(deconv_f), dft_deconv_image)
    print("> Created abs log scaled DFT de-blurred " + dft_deconv_image)
    deconv_s = np.fft.ifft2(deconv_f)
    deconv_s_crop = deconv_s[width:, height:].real
    # Clip
    for y in range(height):
      for x in range(width):
        if(deconv_s_crop[y][x] < 0):
          deconv_s_crop[y][x] = 0
        if(deconv_s_crop[y][x] > 255):
          deconv_s_crop[y][x] = 255
    snr = compute_SNR_in_dB(grayscale_image_array, deconv_s_crop.astype(np.uint8))
    print("SNR after deblurring: " + str(snr) + " dB")
    save_pgm_file(deconv_s_crop.astype(np.uint8), deconv_image)
    print("> Created de-blurred image " + deconv_image)

    #with(np.printoptions(threshold=np.inf)):
    #  print(deconv_s_crop)

  # For each image append the SNRS of each kernel size
  snrs_kernel_image.append(snrs_kernel)

snrs_kernel_image = np.array(snrs_kernel_image)

if(plot):
  legend = input_image_name_arr
  for i, k in enumerate(kernel_sizes):
    plot_snr(limits, snrs_kernel_image[:,i], k, legend)

